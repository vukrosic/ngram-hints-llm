"""Made-up biography facts mixed into FineWeb: web loss and fact recall for published memory methods and ours.

python bio_run.py --arm bag --steps 150

Arms (rows from bio_rows.py):
  none    plain model
  exact   record #367 hints: top next token of the exact last-n table (n = 2,3,4,6) added at input, middle, output
  bag     ours: the same, with order-free tables (set of the last n = 2,3,4 content tokens)
  both    exact + order-free hints
  engram  Engram-style (arXiv 2601.07372, simplified): learned embeddings of hashed 2- and 3-token suffixes,
          added after the first block through a sigmoid gate
  target  CoCoNTs-style (arXiv 2407.02819): the longest exact hint joins the training target, the model sees no hints
  memory  memory layer with product keys (Lample et al. arXiv 1907.05242, Berges et al. arXiv 2412.09764, simplified):
          128 x 128 learned keys, the top 32 of 16,384 learned values mixed by softmax, added after the first block
For the none model the run also scores test-time mixing (infini-gram style): p = (1 - lam s) p_model + lam s [hint],
s = the hint's share, with exact or order-free tables.
Batch = 6 web rows + 2 biography rows. Recall = the right answer is the model's top pick among the possible answers
of that question type (30 cities, 17 universities, 28 companies).
"""
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--arm", default="none", choices=["none", "exact", "bag", "both", "engram", "target", "memory"])
p.add_argument("--steps", type=int, default=150)
p.add_argument("--width", type=int, default=32)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--lr", type=float, default=0.02)
p.add_argument("--hint-lr", type=float, default=0.03)
p.add_argument("--embed-lr", type=float, default=0.003)
p.add_argument("--bio", type=int, default=2, help="biography rows per batch of 8")
p.add_argument("--drop", type=float, default=0.0)
p.add_argument("--target", type=float, default=1.0)
p.add_argument("--buckets", type=int, default=50021)
p.add_argument("--warmup", type=int, default=20)
p.add_argument("--cooldown", type=float, default=0.3)
p.add_argument("--tag", default="")
p.add_argument("--rows", default="bio_rows")
a = p.parse_args()
torch.manual_seed(0)
D = np.load(os.path.join(HERE, "data", a.rows + ".npz"))
GROUPS = ["trained_new_wording", "table_only_new_wording", "table_only_stored_wording"]
LVL = {"exact": [0, 1, 2, 3], "bag": [4, 5, 6], "both": [0, 1, 2, 3, 4, 5, 6], "target": [0, 1, 2, 3]}.get(a.arm, [])
WEB, BIO, VAL = D["web"], D["bio"], D["val"]

parts = [WEB, BIO, VAL, D["web_tok"], D["bio_tok"], D["val_tok"]] + [D[f"{g}_{x}"] for g in GROUPS for x in ("x", "ans", "tok")]
vocab = np.unique(np.concatenate([np.ravel(x) for x in parts])); vocab = vocab[vocab >= 0]; V = len(vocab)


def rm(t):
    t = np.asarray(t); o = np.searchsorted(vocab, t.clip(min=0))
    return torch.as_tensor(np.where(t >= 0, o, -1), dtype=torch.long)


CAND = []
for r in range(3):
    c = np.unique(np.concatenate([D[f"{g}_ans"][r::3] for g in GROUPS])); CAND.append(rm(c))


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


def ngram_ids(x):
    """Engram bucket of the 2- and 3-token suffix ending at each position (raw GPT-2 ids, 0 = too short)."""
    x = np.asarray(x).astype(np.int64); out = []
    for n, mult in [(2, 1000003), (3, 998244353)]:
        h = np.zeros_like(x)
        for i in range(n):
            sh = np.zeros_like(x); sh[:, i:] = x[:, :x.shape[1] - i]; h = (h * mult + sh + 1) % (2 ** 31 - 1)
        h = h % (a.buckets - 1) + 1; h[:, :n - 1] = 0; out.append(torch.as_tensor(h))
    return out


class Memory(nn.Module):
    def __init__(s, d, half=128, top=32, dk=16):
        super().__init__()
        s.top, s.half = top, half
        s.q = nn.Linear(d, 2 * dk, bias=False); s.n = nn.RMSNorm(d)
        s.mem_keys = nn.Parameter(torch.randn(2, half, dk) * dk ** -0.5); s.mem_val = nn.Embedding(half * half, d)
        nn.init.normal_(s.mem_val.weight, std=d ** -0.5)

    def forward(s, x):
        q1, q2 = s.q(s.n(x)).chunk(2, -1)
        v1, i1 = (q1 @ s.mem_keys[0].T).topk(s.top, -1); v2, i2 = (q2 @ s.mem_keys[1].T).topk(s.top, -1)
        sc = (v1[..., :, None] + v2[..., None, :]).flatten(-2); idx = (i1[..., :, None] * s.half + i2[..., None, :]).flatten(-2)
        sc, j = sc.topk(s.top, -1); idx = idx.gather(-1, j)
        return (sc.softmax(-1)[..., None] * s.mem_val(idx)).sum(-2)


class LM(nn.Module):
    def __init__(s):
        super().__init__()
        d = a.width
        s.emb = nn.Embedding(V, d)
        s.blocks = nn.ModuleList(Block(d, a.heads) for _ in range(a.layers))
        s.norm = nn.RMSNorm(d)
        nn.init.normal_(s.emb.weight, std=d ** -0.5)
        nk = max(len(LVL), 1) * 42
        s.sigma = nn.Parameter(torch.zeros(nk)); s.beta = nn.Parameter(torch.zeros(nk, d)); s.alpha = nn.Parameter(torch.ones(3))
        if a.arm == "engram":
            s.gram = nn.ModuleList(nn.Embedding(a.buckets, d) for _ in range(2))
            for e in s.gram:
                nn.init.normal_(e.weight, std=d ** -0.5)
            s.gk, s.gv = nn.Linear(2 * d, d, bias=False), nn.Linear(2 * d, d, bias=False); s.gn1, s.gn2 = nn.RMSNorm(d), nn.RMSNorm(d)
        if a.arm == "memory":
            s.mem = Memory(d)

    def hint(s, tok, kd):
        if tok is None:
            return 0.0
        H = 0.0
        for L in range(tok.shape[0]):
            m = tok[L] >= 0; k = kd[L] + 42 * L
            H = H + (s.sigma[k][..., None] * s.emb(tok[L].clamp(min=0)) + s.beta[k]) * m[..., None]
        return H

    def forward(s, x, tok=None, kd=None, grams=None):
        H = s.hint(tok, kd); h = s.emb(x) + s.alpha[0] * H
        for i, b in enumerate(s.blocks):
            h = b(h)
            if i == 0 and grams is not None:
                e = torch.cat([g(ids) for g, ids in zip(s.gram, grams)], -1)
                gate = torch.sigmoid((s.gn1(h) * s.gn2(s.gk(e))).sum(-1, keepdim=True) / a.width ** 0.5)
                h = h + gate * s.gv(e)
            if i == 0 and a.arm == "memory":
                h = h + s.mem(h)
            if i == len(s.blocks) // 2 - 1:
                h = h + s.alpha[1] * H
        return (s.norm(h) + s.alpha[2] * H) @ s.emb.weight.T


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
hid = [w for n, w in model.named_parameters() if w.ndim == 2 and "emb" not in n and "gram" not in n and "mem_" not in n and n != "beta"]
ids = {id(w) for w in hid}
trust = [model.sigma, model.beta, model.alpha]; tid = {id(w) for w in trust}
rest = [w for w in model.parameters() if id(w) not in ids and id(w) not in tid]
opts = [torch.optim.Adam(rest, lr=a.embed_lr, betas=(0.9, 0.95)), torch.optim.Adam(trust, lr=a.hint_lr, betas=(0.9, 0.95)), Muon(hid, a.lr)]
for o in opts:
    for g in o.param_groups:
        g["base_lr"] = g["lr"]


def lr_scale(step):
    if step < a.warmup:
        return (step + 1) / a.warmup
    cd = int(a.cooldown * a.steps)
    return min(1.0, (a.steps - step) / cd) if cd > 0 else 1.0


def hints(name, rows, on=True):
    if not LVL or a.arm == "target" or not on:
        return None, None
    return rm(D[f"{name}_tok"][LVL][:, rows]), torch.as_tensor(D[f"{name}_kd"][LVL][:, rows].astype(np.int64))


def run(xraw, tok, kd):
    return model(rm(xraw), tok, kd, ngram_ids(xraw) if a.arm == "engram" else None)


SHARE = torch.tensor([0.15, 0.37, 0.62, 0.82, 0.95, 1.0])
nweb = 8 - a.bio; t0 = time.time()
for step in range(a.steps):
    wi = np.arange(step * nweb, (step + 1) * nweb) % len(WEB); bi = np.arange(step * a.bio, (step + 1) * a.bio) % len(BIO)
    xr = np.concatenate([WEB[wi], BIO[bi]]); y = rm(xr[:, 1:])
    tw, kw = hints("web", wi); tb, kb = hints("bio", bi)
    tok = None if tw is None else torch.cat([tw, tb], 1); kd = None if kw is None else torch.cat([kw, kb], 1)
    if tok is not None and a.drop:
        tok = tok.where(torch.rand(tok.shape) > a.drop, torch.tensor(-1))
    if a.arm == "target":
        T = torch.cat([rm(D["web_tok"][LVL][:, wi]), rm(D["bio_tok"][LVL][:, bi])], 1); K = torch.cat([torch.as_tensor(D["web_kd"][LVL][:, wi].astype(np.int64)), torch.as_tensor(D["bio_kd"][LVL][:, bi].astype(np.int64))], 1)
        lp = F.log_softmax(run(xr[:, :-1], None, None), -1); h = torch.full_like(y, -1); sh = torch.zeros(y.shape)
        for L in range(T.shape[0]):
            m = T[L] >= 0; h = torch.where(m, T[L], h); sh = torch.where(m, SHARE[K[L] % 6], sh)
        w = a.target * sh * (h >= 0)
        loss = -((1 - w) * lp.gather(-1, y[..., None])[..., 0] + w * lp.gather(-1, h.clamp(min=0)[..., None])[..., 0]).mean()
    else:
        loss = F.cross_entropy(run(xr[:, :-1], tok, kd).flatten(0, 1), y.flatten())
    loss.backward()
    for o in opts:
        for g in o.param_groups:
            g["lr"] = g["base_lr"] * lr_scale(step)
        o.step()
    model.zero_grad(set_to_none=True)
    if not math.isfinite(loss.item()):
        print("not finite"); break
train_s = time.time() - t0
model.eval()


def longest(name, rows, levels):
    """Longest hint of the given table levels and its share, for test-time mixing."""
    T = rm(D[f"{name}_tok"][levels][:, rows]); K = torch.as_tensor(D[f"{name}_kd"][levels][:, rows].astype(np.int64))
    h = torch.full(T.shape[1:], -1); sh = torch.zeros(T.shape[1:])
    for L in range(T.shape[0]):
        m = T[L] >= 0; h = torch.where(m, T[L], h); sh = torch.where(m, SHARE[K[L] % 6], sh)
    return h, sh


@torch.no_grad()
def web_loss(on):
    tok, kd = hints("val", np.arange(len(VAL)), on)
    return F.cross_entropy(run(VAL[:, :-1], tok, kd).flatten(0, 1), rm(VAL[:, 1:]).flatten()).item()


@torch.no_grad()
def recall(g, lp_fn, mix=None):
    """mix = (levels, lam): p = (1 - lam s) p_model + lam s [hint]."""
    X, pos, ans = D[f"{g}_x"], D[f"{g}_pos"], rm(D[f"{g}_ans"]); r = np.arange(len(X)); pt = torch.as_tensor(pos)
    p = lp_fn(X, r)[torch.as_tensor(r), pt].exp()
    if mix:
        h, sh = longest(g, r, mix[0]); h, sh = h[torch.as_tensor(r), pt], sh[torch.as_tensor(r), pt] * mix[1]
    hit = []
    for i in r:
        c = CAND[i % 3]; sc = p[i, c]
        if mix:
            sc = (1 - sh[i]) * sc + sh[i] * (c == h[i]).float()
        hit.append(c[sc.argmax()].item() == ans[i].item())
    return round(float(np.mean(hit)), 3)


res = dict(vars(a), V=V, params=sum(w.numel() for w in model.parameters()), train_s=round(train_s, 2), last_loss=round(loss.item(), 3))
res["web_hints_off"] = round(web_loss(False), 4)
res["recall_hints_off"] = {g: recall(g, lambda X, r: run(X[:, :-1], None, None)) for g in GROUPS}
if LVL and a.arm != "target":
    res["web_hints_on"] = round(web_loss(True), 4)
    res["recall_hints_on"] = {g: recall(g, lambda X, r, g=g: run(X[:, :-1], *hints(g, r))) for g in GROUPS}
if a.arm == "none":
    with torch.no_grad():
        yv = rm(VAL[:, 1:]); py = F.log_softmax(run(VAL[:, :-1], None, None), -1).gather(-1, yv[..., None])[..., 0].exp()
        for nm, lv in [("exact", [0, 1, 2, 3]), ("bag", [4, 5, 6])]:
            h, sh = longest("val", np.arange(len(VAL)), lv)
            wl = {lam: -torch.log((1 - lam * sh) * py + lam * sh * (h == yv).float() + 1e-9).mean().item() for lam in (0.1, 0.3, 0.5, 0.9)}; lam = min(wl, key=wl.get)
            res[f"mix_{nm}"] = dict(lam=lam, web=round(wl[lam], 4), recall={g: recall(g, lambda X, r: F.log_softmax(run(X[:, :-1], None, None), -1), (lv, lam)) for g in GROUPS},
                                    recall_lam9={g: recall(g, lambda X, r: F.log_softmax(run(X[:, :-1], None, None), -1), (lv, 0.9)) for g in GROUPS})
res["seconds"] = round(time.time() - t0, 2)
print({k: res.get(k) for k in ("arm", "steps", "width", "V", "params", "web_hints_off", "web_hints_on", "recall_hints_off", "recall_hints_on", "mix_exact", "mix_bag", "train_s", "seconds")})
with open(os.path.join(HERE, "bio_results.jsonl"), "a") as f:
    f.write(json.dumps(res) + "\n")
