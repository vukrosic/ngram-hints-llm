"""N-gram hints on FineWeb GPT-2 tokens, using the rows from web_rows.py.

python web_run.py --policy loo --size 7 --drop 0.5 --lr 0.02

Policies: none, seen (earlier steps, as record #367), loo (leave-one-out from corpus index --size),
page (every match from the token's own page left out).
Validation always uses the corpus of --size.
"""
import argparse, glob, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--policy", default="loo", choices=["none", "seen", "loo", "page"])
p.add_argument("--sigma0", type=float, default=0.0, help="starting trust of every hint kind")
p.add_argument("--size", type=int, default=7, help="corpus index, 0 = 0.78M ... 7 = 100M")
p.add_argument("--levels", default="6,10,20")
p.add_argument("--lr", type=float, default=0.02)
p.add_argument("--hint-lr", type=float, default=0.03)
p.add_argument("--embed-lr", type=float, default=0.003)
p.add_argument("--momentum", type=float, default=0.95)
p.add_argument("--steps", type=int, default=200)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--width", type=int, default=64)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--warmup", type=int, default=20)
p.add_argument("--cooldown", type=float, default=0.3)
p.add_argument("--drop", type=float, default=0.0)
p.add_argument("--val-chunks", type=int, default=256)
p.add_argument("--tag", default="")
p.add_argument("--dump", default="", help="save per-token validation losses to data/dump_<name>.npy")
a = p.parse_args()
torch.manual_seed(0)
LV = [int(x) for x in a.levels.split(",")]
R = [np.load(os.path.join(HERE, "data", f"rows_{n}.npz")) for n in LV]
SIZES = R[0]["sizes"].tolist()
used = a.steps * a.batch; nv = a.val_chunks
X, Y = R[0]["x"][:used], R[0]["y"][:used]; VX, VY = R[0]["vx"][:nv], R[0]["vy"][:nv]


def stack(key, size=None):
    return np.stack([(r[key] if size is None else r[key][size]) for r in R])


if a.policy == "seen":
    TT, TK = stack("seen_tok")[:, :used], stack("seen_kd")[:, :used]
elif a.policy in ("loo", "page"):
    TT, TK = stack(a.policy + "_tok", a.size)[:, :used], stack(a.policy + "_kd", a.size)[:, :used]
else:
    TT, TK = None, None
ev = [a.size]
VT = {s: stack("val_tok", s)[:, :nv] for s in ev}; VK = {s: stack("val_kd", s)[:, :nv] for s in ev}

# the model's vocabulary: every token any setting can use, so all runs start from the same weights
parts = [X, Y, VX, VY]
for f in glob.glob(os.path.join(HERE, "data", "rows_*.npz")):
    r = np.load(f); parts += [r["seen_tok"][:used], r["loo_tok"][:, :used], r["page_tok"][:, :used], r["val_tok"][:, :nv]]
vocab = np.unique(np.concatenate([np.ravel(x) for x in parts])); vocab = vocab[vocab >= 0]; V = len(vocab)


def rm(t):
    o = np.searchsorted(vocab, t.clip(min=0))
    return torch.as_tensor(np.where(t >= 0, o, -1), dtype=torch.long)


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
        s.sigma = nn.Parameter(torch.full((nk,), a.sigma0)); s.beta = nn.Parameter(torch.zeros(nk, a.width)); s.alpha = nn.Parameter(torch.ones(3))

    def hint(s, tok, kd):
        if tok is None:
            return 0.0
        H = 0.0
        for L in range(tok.shape[0]):
            m = tok[L] >= 0; k = kd[L] + 42 * L
            H = H + (s.sigma[k][..., None] * s.emb(tok[L].clamp(min=0)) + s.beta[k]) * m[..., None]
        return H

    def forward(s, x, tok=None, kd=None):
        H = s.hint(tok, kd); x = s.emb(x) + s.alpha[0] * H
        for i, b in enumerate(s.blocks):
            x = b(x)
            if i == len(s.blocks) // 2 - 1:
                x = x + s.alpha[1] * H
        return (s.norm(x) + s.alpha[2] * H) @ s.emb.weight.T


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
trust = [model.sigma, model.beta, model.alpha]; tid = {id(w) for w in trust}
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


@torch.no_grad()
def val(tok, kd):
    """Loss over all validation tokens, and over the tokens that have a hint."""
    model.eval(); tot, hl, hn, n = 0.0, 0.0, 0, 0
    for i in range(0, nv, 32):
        x, y = rm(VX[i:i + 32]), rm(VY[i:i + 32])
        tk, kk = (None, None) if tok is None else (rm(tok[:, i:i + 32]), torch.as_tensor(kd[:, i:i + 32].astype(np.int64)))
        l = F.cross_entropy(model(x, tk, kk).flatten(0, 1), y.flatten(), reduction="none")
        tot += l.sum().item(); n += l.numel()
        if tk is not None:
            m = (tk >= 0).any(0).flatten(); hl += l[m].sum().item(); hn += m.sum().item()
    model.train()
    return tot / n, hl / max(hn, 1), hn / n


t0 = time.time()
for step in range(a.steps):
    i = np.arange(step * a.batch, (step + 1) * a.batch)
    x, y = rm(X[i]), rm(Y[i])
    if TT is None:
        tok, kd = None, None
    else:
        tok, kd = rm(TT[:, i]), torch.as_tensor(TK[:, i].astype(np.int64))
        if a.drop:
            tok = tok.where(torch.rand(tok.shape) > a.drop, torch.tensor(-1))
    loss = F.cross_entropy(model(x, tok, kd).flatten(0, 1), y.flatten())
    loss.backward()
    for o in opts:
        for g in o.param_groups:
            g["lr"] = g["base_lr"] * lr_scale(step)
        o.step()
    model.zero_grad(set_to_none=True)
    if not math.isfinite(loss.item()):
        print("not finite"); break
train_s = time.time() - t0
res = dict(vars(a), V=V, params=sum(w.numel() for w in model.parameters()), tokens=a.steps * a.batch * 64, train_s=round(train_s, 2), last_loss=round(loss.item(), 3))
res["val_nohint"] = round(val(None, None)[0], 4)
if a.policy != "none":
    res["curve"] = [[SIZES[s], *[round(v, 4) for v in val(VT[s], VK[s])]] for s in ev]
    res["trust"] = [round(x.item(), 2) for x in model.alpha.detach()]
if a.dump:
    @torch.no_grad()
    def token_losses(tok, kd):
        model.eval(); out = []
        for i in range(0, nv, 32):
            tk, kk = (None, None) if tok is None else (rm(tok[:, i:i + 32]), torch.as_tensor(kd[:, i:i + 32].astype(np.int64)))
            out.append(F.cross_entropy(model(rm(VX[i:i + 32]), tk, kk).flatten(0, 1), rm(VY[i:i + 32]).flatten(), reduction="none").view(-1, VX.shape[1]))
        return torch.cat(out).numpy()

    np.save(os.path.join(HERE, "data", f"dump_{a.dump}.npy"), token_losses(*((VT[a.size], VK[a.size]) if TT is not None else (None, None))))
res["seconds"] = round(time.time() - t0, 2)
print({k: res.get(k) for k in ("policy", "size", "drop", "lr", "V", "params", "last_loss", "val_nohint", "curve", "train_s", "seconds")})
with open(os.path.join(HERE, "web_results.jsonl"), "a") as f:
    f.write(json.dumps(res) + "\n")
