"""How often must a fact appear before a model knows it without hints?

python rare.py --hints 1 --width 64 --repeat 1

Each fact is stated an exact number of times in training (1, 2, 4 ... 256, times --repeat). The table holds every fact.
"""
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--hints", type=int, default=1)
p.add_argument("--repeat", type=int, default=1)
p.add_argument("--lr", type=float, default=0.02)
p.add_argument("--hint-lr", type=float, default=0.1)
p.add_argument("--embed-lr", type=float, default=0.003)
p.add_argument("--momentum", type=float, default=0.95)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--width", type=int, default=64)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--warmup", type=int, default=20)
p.add_argument("--cooldown", type=float, default=0.3)
p.add_argument("--tag", default="")
a = p.parse_args()
torch.manual_seed(0)
NAMES = 100; EXTRA = 3; PER = 200; COUNTS = [2 ** i for i in range(9)]; LV = [2, 3, 4, 6]; SITES = {0, 1, 2}; PAD = 0; POS = 6
THE, OF, IS, DOT = range(NAMES + 1, NAMES + 5); REL = NAMES + 5; V = REL + 4; VW = 0
BASE = np.uint64(1000003); C_EDGES = np.array([1, 2, 4, 8, 16, 64]); P_EDGES = np.array([0.25, 0.5, 0.75, 0.9, 0.9999])
FACT = torch.arange(V)
rng = np.random.default_rng(0)
ids = rng.choice(4_000_000, PER * len(COUNTS), replace=False)
F_ = np.stack([ids % 4, ids // 4 % NAMES + 1, ids // 400 % NAMES + 1, ids // 40000 % NAMES + 1, rng.integers(1, NAMES + 1, len(ids)), rng.integers(1, NAMES + 1, len(ids))], 1)
n = len(F_); k = lambda t: np.full(n, t)
L = np.stack([k(THE), F_[:, 0] + REL, k(OF), F_[:, 1], F_[:, 2], F_[:, 3], k(IS), F_[:, 4], F_[:, 5], k(DOT)], 1).astype(np.int64)
times = np.repeat(np.array(COUNTS) * a.repeat, PER)
stream = L[np.repeat(np.arange(n), times)]; stream = stream[rng.permutation(len(stream))]
corpus = np.concatenate([stream, np.repeat(L, EXTRA, 0)])
steps = len(stream) // a.batch


def mix(k):
    k = k ^ (k >> np.uint64(29)); k = k * np.uint64(0xbf58476d1ce4e5b9); k = k ^ (k >> np.uint64(32))
    return k & np.uint64(0xffffffffffff0000)


def keys(X, m):
    h = np.zeros(X.shape, dtype=np.uint64); pw = np.uint64(1)
    for i in range(m):
        h[:, m - 1:] += X[:, m - 1 - i:X.shape[1] - i].astype(np.uint64) * pw; pw = pw * BASE
    o = mix(h); o[:, :m - 1] = 0
    return o


def kind(c, t):
    return np.searchsorted(C_EDGES, c, side="right") * 6 + np.searchsorted(P_EDGES, t / c, side="left")


def table(m):
    X, Y = corpus[:, :-1], corpus[:, 1:]; q = keys(X, m); keep = q != 0
    u, cnt = np.unique(q[keep] | Y[keep].astype(np.uint64), return_counts=True); uk = u & np.uint64(0xffffffffffff0000)
    first = np.flatnonzero(np.r_[True, uk[1:] != uk[:-1]])
    sc = (cnt.astype(np.uint64) << np.uint64(16)) | (u & np.uint64(0xffff))
    top = np.maximum.reduceat(sc, first); tot = np.add.reduceat(cnt, first)
    sc[sc == np.repeat(top, np.diff(np.r_[first, len(sc)]))] = 0
    sec = np.maximum.reduceat(sc, first)
    return uk[first], (top & np.uint64(0xffff)).astype(np.int64), (top >> np.uint64(16)).astype(np.int64), (sec & np.uint64(0xffff)).astype(np.int64), (sec >> np.uint64(16)).astype(np.int64), tot.astype(np.int64)


def lookup(t, M, loo):
    K, t1, n1, t2, n2, tot = t; X, Y = M[:, :-1], M[:, 1:]; q = keys(X, m_)
    i = np.searchsorted(K, q).clip(0, len(K) - 1); hit = (K[i] == q) & (q != 0)
    if loo:
        c = tot[i] - 1; a1 = t1[i].copy(); b = n1[i].copy()
        is1 = a1 == Y; b = np.where(is1, b - 1, b); use2 = is1 & (n2[i] > b)
        a1 = np.where(use2, t2[i], a1); b = np.where(use2, n2[i], b); hit &= c > 0
    else:
        c = tot[i]; a1 = t1[i]; b = n1[i]
    tok = np.where(hit, a1, -1); kd = np.zeros_like(tok); kd[hit] = kind(c[hit], np.maximum(b[hit], 1))
    return tok, kd


TT, TK, GT, GK = [], [], [], []
for m_ in LV:
    t = table(m_); x, y = lookup(t, stream, True); TT.append(x); TK.append(y); x, y = lookup(t, L, False); GT.append(x); GK.append(y)
TT, TK, GT, GK = (torch.as_tensor(np.stack(v)) for v in (TT, TK, GT, GK))
S = torch.as_tensor(stream); LT = torch.as_tensor(L)


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

    def hint(s, tok, kd):
        if tok is None:
            return 0.0
        H = 0.0
        for L in range(tok.shape[0]):
            m = tok[L] >= 0; k = kd[L] + 42 * L
            H = H + (s.sigma[k][..., None] * s.emb(tok[L].clamp(min=0)) + s.beta[k]) * m[..., None]
        return H

    def forward(s, x, tok=None, kd=None, part=None):
        H = s.hint(tok, kd); use = lambda site: s.alpha[site] * H if site in SITES else 0.0
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
hid = [w for n_, w in model.named_parameters() if w.ndim == 2 and "emb" not in n_ and n_ != "beta"]
ids_ = {id(w) for w in hid}
trust = [model.sigma, model.beta, model.alpha]; tid = {id(w) for w in trust}
rest = [w for w in model.parameters() if id(w) not in ids_ and id(w) not in tid]
opts = [torch.optim.Adam(rest, lr=a.embed_lr, betas=(0.9, 0.95)), torch.optim.Adam(trust, lr=a.hint_lr, betas=(0.9, 0.95)), Muon(hid, a.lr, a.momentum)]
for o in opts:
    for g in o.param_groups:
        g["base_lr"] = g["lr"]


def lr_scale(step):
    if step < a.warmup:
        return (step + 1) / a.warmup
    cd = int(a.cooldown * steps)
    return min(1.0, (steps - step) / cd) if cd > 0 else 1.0


@torch.no_grad()
def acc(hints):
    model.eval(); x, y = LT[:, :-1], LT[:, 1:]; out = []
    for i in range(0, n, 256):
        out.append(model(x[i:i + 256], GT[:, i:i + 256] if hints else None, GK[:, i:i + 256] if hints else None, FACT)[:, POS].argmax(-1))
    model.train(); ok = (torch.cat(out) == y[:, POS]).float()
    return [round(ok[j * PER:(j + 1) * PER].mean().item(), 3) for j in range(len(COUNTS))]


t0 = time.time()
for step in range(steps):
    i = torch.arange(step * a.batch, (step + 1) * a.batch); b = S[i]; x, y = b[:, :-1], b[:, 1:]
    tok, kd = (TT[:, i], TK[:, i]) if a.hints else (None, None)
    loss = F.cross_entropy(model(x, tok, kd, FACT).flatten(0, 1), y.flatten(), ignore_index=PAD)
    loss.backward()
    for o in opts:
        for g in o.param_groups:
            g["lr"] = g["base_lr"] * lr_scale(step)
        o.step()
    model.zero_grad(set_to_none=True)
    if not math.isfinite(loss.item()):
        print("not finite"); break
res = dict(vars(a), steps=steps, params=sum(w.numel() for w in model.parameters()), counts=[c * a.repeat for c in COUNTS], on=acc(True) if a.hints else None, off=acc(False), last_loss=round(loss.item(), 3))
res["seconds"] = round(time.time() - t0, 2)
print(a.hints, a.width, a.repeat, res["on"], res["off"], res["seconds"], flush=True)
with open(os.path.join(HERE, "rare_results.jsonl"), "a") as f:
    f.write(json.dumps(res) + "\n")
