"""N-gram hints on the Muon lab model: a hint corpus says which character most often followed the last n characters.

python shakespeare.py --policy loo --width 64 --steps 200 --lr 0.02

Hint policies during training (validation always uses the whole training split as the corpus):
  none  no hints at all
  seen  strictly causal: a step's hints count only tokens trained in earlier steps (what record #367 does)
  self  whole training split, each token's own occurrence still inside the counts (leaks the answer)
  loo   whole training split, each token's own occurrence removed from the counts (leave-one-out)
Training streams non-overlapping shuffled chunks, so each token is trained at most once per pass.
"""
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--policy", default="loo", choices=["none", "seen", "self", "loo"])
p.add_argument("--levels", default="6,12,24")
p.add_argument("--lr", type=float, default=0.02)
p.add_argument("--hint-lr", type=float, default=0.03)
p.add_argument("--embed-lr", type=float, default=0.003)
p.add_argument("--momentum", type=float, default=0.95)
p.add_argument("--steps", type=int, default=200)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--seq", type=int, default=64)
p.add_argument("--width", type=int, default=64)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--warmup", type=int, default=20)
p.add_argument("--cooldown", type=float, default=0.3)
p.add_argument("--drop", type=float, default=0.0, help="drop each training hint with this probability")
p.add_argument("--curve", type=int, default=0, help="1: evaluate the final model with doubling corpus sizes")
p.add_argument("--tag", default="")
a = p.parse_args()
torch.manual_seed(0); rng = np.random.default_rng(0)
LV = [int(x) for x in a.levels.split(",")] if a.policy != "none" else []

text = open(os.path.join(HERE, "data", "shakespeare.txt"), encoding="utf-8").read()
chars = sorted(set(text)); V = len(chars)
S = np.array([chars.index(c) for c in text], dtype=np.int64) if False else np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)
remap = -np.ones(256, dtype=np.int64); remap[np.unique(S)] = np.arange(len(np.unique(S))); S = remap[S]; V = int(S.max()) + 1
CUT = int(0.9 * len(S)); N = len(S)
C_EDGES = np.array([1, 2, 4, 8, 16, 64]); P_EDGES = np.array([0.25, 0.5, 0.75, 0.9, 0.9999])  # 7 count, 6 purity buckets


def keys(n):
    """keys[p] = hash of S[p-n:p], the n characters before position p."""
    w = np.lib.stride_tricks.sliding_window_view(S, n).astype(np.uint64)
    pw = np.array([pow(1000003, n - 1 - i, 2**64) for i in range(n)], dtype=np.uint64)
    k = np.zeros(N, dtype=np.uint64); k[n:] = (w[:-1] * pw).sum(1)
    return k


KEYS = {n: keys(n) for n in LV or [6, 12, 24]}


def build(n, end, sel=None):
    """Tables over corpus positions n..end-1 (or positions sel): for each n-gram its total and its two most common next characters."""
    k, c = (KEYS[n][n:end], S[n:end]) if sel is None else (KEYS[n][sel], S[sel])
    o = np.lexsort((c, k)); k, c = k[o], c[o]
    new = np.r_[True, (k[1:] != k[:-1]) | (c[1:] != c[:-1])]
    idx = np.flatnonzero(new); cnt = np.diff(np.r_[idx, len(k)]); uk, uc = k[idx], c[idx]
    o = np.lexsort((-cnt, uk)); uk, uc, cnt = uk[o], uc[o], cnt[o]
    first = np.flatnonzero(np.r_[True, uk[1:] != uk[:-1]])
    total = np.add.reduceat(cnt, first)
    has2 = np.r_[first[1:], len(uk)] - first > 1
    c2 = np.where(has2, uc[np.minimum(first + 1, len(uc) - 1)], -1)
    n2 = np.where(has2, cnt[np.minimum(first + 1, len(cnt) - 1)], 0)
    return dict(key=uk[first], t1=uc[first], n1=cnt[first], t2=c2, n2=n2, tot=total)


def kind(count, top):
    return np.searchsorted(C_EDGES, count, side="right") * 6 + np.searchsorted(P_EDGES, top / count, side="left")


def lookup(tb, n, pos, loo=False):
    """Hint token (-1 = none) and kind for target positions pos."""
    k = KEYS[n][pos]; i = np.searchsorted(tb["key"], k).clip(0, len(tb["key"]) - 1)
    hit = (tb["key"][i] == k) & (pos >= n)
    t, cnt, top = tb["t1"][i].copy(), tb["tot"][i].astype(np.int64), tb["n1"][i].astype(np.int64)
    if loo:
        own = S[pos]; cnt = cnt - 1
        is1 = t == own
        top = np.where(is1, top - 1, top)
        use2 = is1 & (tb["n2"][i] > top)
        t = np.where(use2, tb["t2"][i], t); top = np.where(use2, tb["n2"][i], top)
        hit &= cnt > 0
    tok = np.where(hit, t, -1)
    kd = np.zeros_like(tok); kd[hit] = kind(cnt[hit], np.maximum(top[hit], 1))
    return tok, kd


TABLES = {n: build(n, CUT) for n in LV}


class Stream:
    """Online counts for the 'seen' policy: only tokens from earlier steps."""
    def __init__(s): s.d = {n: {} for n in LV}

    def get(s, n, pos):
        tok, kd = -np.ones_like(pos), np.zeros_like(pos); D = s.d[n]
        for i, (q, k) in enumerate(zip(pos.ravel().tolist(), KEYS[n][pos].ravel().tolist())):
            e = D.get(k)
            if e and q >= n:
                c = max(e, key=e.get); tot = sum(e.values())
                tok.flat[i] = c; kd.flat[i] = int(kind(np.array([tot]), np.array([e[c]]))[0])
        return tok, kd

    def add(s, pos):
        for n in LV:
            D = s.d[n]
            for k, c in zip(KEYS[n][pos].ravel().tolist(), S[pos].ravel().tolist()):
                e = D.setdefault(k, {}); e[c] = e.get(c, 0) + 1


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
        nk = max(len(LV), 1) * 42
        s.sigma = nn.Parameter(torch.zeros(nk)); s.beta = nn.Parameter(torch.zeros(nk, a.width)); s.alpha = nn.Parameter(torch.ones(3))

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


L = a.seq + 1
starts = rng.permutation((CUT - L) // L) * L
vs = np.arange(CUT, N - L - 1, L)[rng.permutation((N - CUT - L) // L - 1)[:96]]


def hints_for(pos, tables, loo=False, stream=None):
    if not LV or (tables is None and stream is None):
        return None, None
    out = [stream.get(n, pos) if stream else lookup(tables[n], n, pos, loo) for n in LV]
    return torch.as_tensor(np.stack([o[0] for o in out])), torch.as_tensor(np.stack([o[1] for o in out]))


@torch.no_grad()
def val(tables, hintless=False):
    """Validation loss; also the loss on tokens with any hint and the share of such tokens."""
    model.eval(); tot, hit_l, hit_n, n = 0.0, 0.0, 0, 0
    for i in range(0, len(vs), 32):
        p0 = vs[i:i + 32]; pos = p0[:, None] + np.arange(1, L)
        x = torch.as_tensor(S[pos - 1]); y = torch.as_tensor(S[pos])
        tok, kd = hints_for(pos, None if hintless else tables)
        l = F.cross_entropy(model(x, tok, kd).flatten(0, 1), y.flatten(), reduction="none")
        tot += l.sum().item(); n += l.numel()
        if tok is not None:
            m = (tok >= 0).any(0).flatten(); hit_l += l[m].sum().item(); hit_n += m.sum().item()
    model.train()
    return tot / n, hit_l / max(hit_n, 1), hit_n / n


stream = Stream() if a.policy == "seen" else None
tables = TABLES if a.policy in ("self", "loo") else None
t0 = time.time(); I = 0
for step in range(a.steps):
    if I + a.batch > len(starts):
        starts = rng.permutation(starts); I = 0
    p0 = starts[I:I + a.batch]; I += a.batch
    pos = p0[:, None] + np.arange(1, L)
    x, y = torch.as_tensor(S[pos - 1]), torch.as_tensor(S[pos])
    if a.policy == "none":
        tok, kd = None, None
    else:
        tok, kd = hints_for(pos, tables, a.policy == "loo", stream)
        if a.drop:
            tok = tok.where(torch.rand(tok.shape) > a.drop, torch.tensor(-1))
    loss = F.cross_entropy(model(x, tok, kd).flatten(0, 1), y.flatten())
    loss.backward()
    if stream:
        stream.add(pos)
    for o in opts:
        for g in o.param_groups:
            g["lr"] = g["base_lr"] * lr_scale(step)
        o.step()
    model.zero_grad(set_to_none=True)
    if not math.isfinite(loss.item()):
        print("not finite"); break
secs = time.time() - t0
res = dict(vars(a), params=sum(w.numel() for w in model.parameters()), seconds=round(secs, 2), tokens=a.steps * a.batch * a.seq)
res["val_nohint"] = round(val(None, True)[0], 4)
if LV:
    v, vh, share = val(TABLES); res.update(val=round(v, 4), val_hinted=round(vh, 4), hint_share=round(share, 4))
    res["trust"] = [round(x.item(), 2) for x in model.alpha.detach()]
if a.curve and LV:
    cv = []
    perm = np.random.default_rng(5).permutation((CUT - 100) // 256)  # random 256-character pieces of the training split
    for e in [CUT // 2 ** i for i in range(7, -1, -1)]:
        sel = (perm[:e // 256, None] * 256 + 100 + np.arange(256)).ravel()
        sel = sel[sel < CUT]
        tb = {n: build(n, e, sel) for n in LV} if a.curve == 2 else {n: build(n, e) for n in LV}
        v, vh, share = val(tb); cv.append([int(e), round(v, 4), round(vh, 4), round(share, 4)])
    res["curve"] = cv
print({k: res.get(k) for k in ("policy", "width", "steps", "lr", "drop", "val", "val_hinted", "hint_share", "val_nohint", "seconds", "params", "curve")})
with open(os.path.join(HERE, "results.jsonl"), "a") as f:
    f.write(json.dumps(res) + "\n")
