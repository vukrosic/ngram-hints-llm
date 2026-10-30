"""One model trains on web text and on the made-up language of facts.py at once.

python mixed.py --sites 2 --drop 0.5 --width 64

Run facts.py and web_rows.py first; they write the rows this file reads. The two languages share no tokens, so one
joined hint table gives the same hints as these two tables. Each language's loss covers only its own tokens, for speed.
"""
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--hints", type=int, default=1)
p.add_argument("--sites", default="0,1,2", help="where hints enter: 0 input, 1 middle, 2 output")
p.add_argument("--drop", type=float, default=0.0, help="share of hints dropped in training")
p.add_argument("--drop-all", type=float, default=0.0, help="chance a token gets no hint at all in training")
p.add_argument("--aux", type=int, default=0, help="1 = every step also trains on the same batch without hints")
p.add_argument("--right", type=int, default=0, help="1 = a switch on each hint, trained to predict whether the hint is right; needs --aux 1")
p.add_argument("--max-seen", type=int, default=0, help="tables keep only contexts in count buckets below this, in training and at evaluation")
p.add_argument("--lr", type=float, default=0.02)
p.add_argument("--hint-lr", type=float, default=0.1)
p.add_argument("--embed-lr", type=float, default=0.003)
p.add_argument("--momentum", type=float, default=0.95)
p.add_argument("--steps", type=int, default=5760)
p.add_argument("--batch", type=int, default=32, help="made-up language lines per step")
p.add_argument("--web-batch", type=int, default=2, help="web chunks per step")
p.add_argument("--width", type=int, default=64)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--warmup", type=int, default=20)
p.add_argument("--cooldown", type=float, default=0.3)
p.add_argument("--val-chunks", type=int, default=256)
p.add_argument("--tag", default="")
a = p.parse_args()
torch.manual_seed(0)
SITES = {int(c) for c in a.sites.split(",")}
LV = [2, 3, 4, 5, 6, 10, 20]; FACT_LV = [2, 3, 4, 6]; SIZE = 7; PAD = 0

W = [np.load(os.path.join(HERE, "data", f"rows_{n}.npz")) for n in LV]
uw = a.steps * a.web_batch; nv = a.val_chunks
WX, WY, VX, VY = W[0]["x"][:uw], W[0]["y"][:uw], W[0]["vx"][:nv], W[0]["vy"][:nv]
WT = np.stack([r["page_tok"][SIZE][:uw] for r in W]); WK = np.stack([r["page_kd"][SIZE][:uw] for r in W])
VT = np.stack([r["val_tok"][SIZE][:nv] for r in W]); VK = np.stack([r["val_kd"][SIZE][:nv] for r in W])
vocab = np.unique(np.concatenate([np.ravel(x) for x in (WX, WY, VX, VY, WT, VT)])); vocab = vocab[vocab >= 0]; VW = len(vocab)


def rm(t):
    o = np.searchsorted(vocab, t.clip(min=0))
    return torch.as_tensor(np.where(t >= 0, o, -1), dtype=torch.long)


D = np.load(os.path.join(HERE, "data", "facts_rows.npz")); ut = a.steps * a.batch
VF = int(D["stream"].max()) + 1; V = VW + VF
GROUPS = ["common_fact", "rare_table", "rare_other_words", "fact_not_in_table", "seen_sum", "held_out_sum", "read_fresh", "chain_fresh"]
WEB, FACT = torch.arange(VW), torch.arange(VW, V)


def seven(t):
    """Facts have lengths 2, 3, 4, 6; the lengths they lack get no hint."""
    o = np.full((len(LV), *t.shape[1:]), -1, dtype=np.int64)
    for i, n in enumerate(FACT_LV):
        o[LV.index(n)] = t[i]
    return o


def shift(t):
    return torch.as_tensor(np.where(t >= 0, t + VW, -1), dtype=torch.long)


stream = torch.as_tensor(D["stream"][:ut]); TT = shift(seven(D["tt"][:, :ut])); TK = torch.as_tensor(seven(D["tk"][:, :ut])).clamp(min=0)
WXt, WYt, WTt, WKt = rm(WX), rm(WY), rm(WT), torch.as_tensor(WK.astype(np.int64))
if a.max_seen:
    WTt = WTt.where(WKt // 6 < a.max_seen, torch.tensor(-1)); VT = np.where(VK // 6 < a.max_seen, VT, -1); TT = TT.where(TK // 6 < a.max_seen, torch.tensor(-1))


def rope(x, base=10000.0):
    t, d = x.shape[-2], x.shape[-1]
    f = 1.0 / base ** (torch.arange(0, d, 2).float() / d)
    ang = torch.outer(torch.arange(t).float(), f); c, s = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x2 * c + x1 * s], -1).flatten(-2)


class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.h, s.dk = h, d // h
        s.q, s.k, s.v, s.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        hid = int(8 * d / 3)
        s.gate, s.up, s.down = nn.Linear(d, hid, bias=False), nn.Linear(d, hid, bias=False), nn.Linear(hid, d, bias=False)
        s.n1, s.n2, s.qn, s.kn = nn.RMSNorm(d), nn.RMSNorm(d), nn.RMSNorm(s.dk), nn.RMSNorm(s.dk)

    def forward(s, x):
        b, t, d = x.shape
        h = s.n1(x)
        q, k, v = (m(h).view(b, t, s.h, s.dk).transpose(1, 2) for m in (s.q, s.k, s.v))
        q, k = rope(s.qn(q)), rope(s.kn(k))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.o(y.transpose(1, 2).reshape(b, t, d))
        h = s.n2(x)
        return x + s.down(F.silu(s.gate(h)) * s.up(h))


class LM(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(V, a.width)
        s.blocks = nn.ModuleList(Block(a.width, a.heads) for _ in range(a.layers))
        s.norm = nn.RMSNorm(a.width)
        nn.init.normal_(s.emb.weight, std=a.width ** -0.5)
        nk = len(LV) * 42
        s.sigma = nn.Parameter(torch.zeros(nk)); s.beta = nn.Parameter(torch.zeros(nk, a.width)); s.alpha = nn.Parameter(torch.ones(3))
        s.sw = nn.Parameter(torch.zeros(a.width)); s.sw0 = nn.Parameter(torch.tensor(2.0))

    def hint(s, tok, kd):
        if tok is None:
            return 0.0
        H = 0.0
        for L in range(tok.shape[0]):
            m = tok[L] >= 0; k = kd[L] + 42 * L
            H = H + (s.sigma[k][..., None] * s.emb(tok[L].clamp(min=0)) + s.beta[k]) * m[..., None]
        return H

    def plain(s, x):
        h = s.emb(x)
        for b in s.blocks:
            h = b(h)
        return s.norm(h)

    def forward(s, x, tok=None, kd=None, part=None):
        H = s.hint(tok, kd); g = 1.0
        if a.right and tok is not None:
            s.h0 = s.plain(x); s.logit = s.h0.detach() @ s.sw + s.sw0; g = s.logit.sigmoid()[..., None]
        use = lambda site: g * s.alpha[site] * H if site in SITES else 0.0
        x = s.emb(x) + use(0)
        for i, b in enumerate(s.blocks):
            x = b(x)
            if i == len(s.blocks) // 2 - 1:
                x = x + use(1)
        return (s.norm(x) + use(2)) @ s.emb.weight[part].T


def newton_schulz(G, steps=5):
    X = G / (G.norm() + 1e-7)
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    return X.T if tall else X


class Muon(torch.optim.Optimizer):
    def __init__(s, params, lr, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(s):
        for g in s.param_groups:
            for w in g["params"]:
                buf = s.state[w].setdefault("buf", torch.zeros_like(w))
                buf.mul_(g["momentum"]).add_(w.grad)
                u = newton_schulz(w.grad.add(buf, alpha=g["momentum"]))
                u = u * max(1.0, w.shape[0] / w.shape[1]) ** 0.5
                w.add_(u, alpha=-g["lr"])


model = LM()
hid = [w for n, w in model.named_parameters() if w.ndim == 2 and "emb" not in n and n != "beta"]
ids = {id(w) for w in hid}
trust = [model.sigma, model.beta, model.alpha, model.sw, model.sw0]; tid = {id(w) for w in trust}
rest = [w for w in model.parameters() if id(w) not in ids and id(w) not in tid]
opts = [torch.optim.Adam(rest, lr=a.embed_lr, betas=(0.9, 0.95)), torch.optim.Adam(trust, lr=a.hint_lr, betas=(0.9, 0.95)), Muon(hid, a.lr, a.momentum)]
for o in opts:
    for g in o.param_groups:
        g["base_lr"] = g["lr"]


def lr_scale(step):
    if step < a.warmup:
        return (step + 1) / a.warmup
    cd = int(a.cooldown * a.steps)
    return min(1.0, (a.steps - step) / cd) if cd > 0 else 1.0


def dropped(tok):
    tok = tok.where(torch.rand(tok.shape) > a.drop, torch.tensor(-1)) if a.drop else tok
    return tok.where(torch.rand(tok.shape[1:]) > a.drop_all, torch.tensor(-1)) if a.drop_all else tok


def right_loss(tok, y):
    """The switch learns whether the longest hint equals the next token."""
    has = (tok >= 0).any(0) & (y != PAD); last = tok[(tok >= 0).int().cumsum(0).argmax(0), torch.arange(len(y))[:, None], torch.arange(y.shape[1])]
    return F.binary_cross_entropy_with_logits(model.logit[has], (last[has] == y[has]).float())


@torch.no_grad()
def web_val(hints):
    model.eval(); tot, n = 0.0, 0
    sw, ns = 0.0, 0
    for i in range(0, nv, 32):
        x, y = rm(VX[i:i + 32]), rm(VY[i:i + 32])
        tk, kk = (rm(VT[:, i:i + 32]), torch.as_tensor(VK[:, i:i + 32].astype(np.int64))) if hints else (None, None)
        l = F.cross_entropy(model(x, tk, kk, WEB).flatten(0, 1), y.flatten(), reduction="sum"); tot += l.item(); n += y.numel()
        if hints and a.right:
            m = (tk >= 0).any(0); sw += model.logit[m].sigmoid().sum().item(); ns += m.sum().item()
    model.train()
    if ns:
        print("web switch mean", round(sw / ns, 3), flush=True)
    return round(tot / n, 4)


@torch.no_grad()
def task(g):
    m = torch.as_tensor(D[f"{g}_x"]); pos = int(D[f"{g}_pos"]); x, y = m[:, :-1], m[:, 1:]
    tok = shift(seven(D[f"{g}_tok"])); kd = torch.as_tensor(seven(D[f"{g}_kd"])).clamp(min=0)
    tok = tok.where(kd // 6 < a.max_seen, torch.tensor(-1)) if a.max_seen else tok
    x = x + VW; model.eval()
    on = model(x, tok, kd, FACT) if a.hints else model(x, None, None, FACT)
    sw = round(model.logit[:, pos].sigmoid().mean().item(), 3) if a.right else None
    off = model(x, None, None, FACT)
    model.train()
    return dict(acc=round((on[:, pos].argmax(-1) == y[:, pos]).float().mean().item(), 3), acc_off=round((off[:, pos].argmax(-1) == y[:, pos]).float().mean().item(), 3), switch=sw)


t0 = time.time()
for step in range(a.steps):
    i = torch.arange(step * a.batch, (step + 1) * a.batch); j = torch.arange(step * a.web_batch, (step + 1) * a.web_batch)
    b = stream[i]; x, y = b[:, :-1] + VW, b[:, 1:]
    t1, k1, t2, k2 = (dropped(TT[:, i]), TK[:, i], dropped(WTt[:, j]), WKt[:, j]) if a.hints else (None, None, None, None)
    lf = F.cross_entropy(model(x, t1, k1, FACT).flatten(0, 1), y.flatten(), ignore_index=PAD)
    bf = right_loss(t1, y + VW) if a.right else 0.0
    lw = F.cross_entropy(model(WXt[j], t2, k2, WEB).flatten(0, 1), WYt[j].flatten())
    bw = right_loss(t2, WYt[j]) if a.right else 0.0
    if a.aux:
        lf = (lf + F.cross_entropy(model(x, None, None, FACT).flatten(0, 1), y.flatten(), ignore_index=PAD)) / 2
        lw = (lw + F.cross_entropy(model(WXt[j], None, None, WEB).flatten(0, 1), WYt[j].flatten())) / 2
    loss = (lf + lw + bf + bw) / 2
    loss.backward()
    for o in opts:
        for g in o.param_groups:
            g["lr"] = g["base_lr"] * lr_scale(step)
        o.step()
    model.zero_grad(set_to_none=True)
    if not math.isfinite(loss.item()):
        print("not finite"); break
train_s = time.time() - t0
res = dict(vars(a), V=V, params=sum(w.numel() for w in model.parameters()), train_s=round(train_s, 2), web_off=web_val(False), web_on=web_val(True) if a.hints else None, groups={g: task(g) for g in GROUPS})
res["seconds"] = round(time.time() - t0, 2)
print(res["hints"], a.sites, a.drop, res["web_on"], res["web_off"], res["seconds"], flush=True)
with open(os.path.join(HERE, "mixed_results.jsonl"), "a") as f:
    f.write(json.dumps(res) + "\n")
