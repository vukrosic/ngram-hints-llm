"""Do n-gram hints help facts or reasoning? A made-up language, so every answer has a known kind.

python facts.py --hints 1 --steps 600 --lr 0.02

Each line is one sentence. The table counts next tokens over 1.5M lines; the model trains on a small part of them.
Training hints leave out the line's own occurrence. Test lines never enter training; only the facts of the
"rare_table" and "rare_other_words" groups are in the table.
"""
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "facts_rows.npz")
p = argparse.ArgumentParser()
p.add_argument("--hints", type=int, default=1)
p.add_argument("--lr", type=float, default=0.02)
p.add_argument("--hint-lr", type=float, default=0.1)
p.add_argument("--embed-lr", type=float, default=0.003)
p.add_argument("--momentum", type=float, default=0.95)
p.add_argument("--steps", type=int, default=600)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--width", type=int, default=64)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--warmup", type=int, default=20)
p.add_argument("--cooldown", type=float, default=0.3)
p.add_argument("--tag", default="")
p.add_argument("--drop", type=float, default=0.0, help="share of hints dropped in training")
p.add_argument("--sites", default="0,1,2", help="where hints enter: 0 input, 1 middle, 2 output")
p.add_argument("--use", default="0,1,2,3", help="which hint lengths to use, as indices of LEVELS")
a = p.parse_args()
torch.manual_seed(0)
SITES = {int(c) for c in a.sites.split(",")}; USE = [int(c) for c in a.use.split(",")]

NL = 1_500_000; POOL = 184_320; NTEST = 1000; T = 18; LEVELS = [2, 3, 4, 6]; NAMES = 100
PAD = 0; DIG = NAMES + 1; W = DIG + 10
THE, OF, IS, FOR, WAS, PLUS, EQ, DOT, TALLER, THAN, WHO, TALLEST, Q, OWNS, WHAT, DOES, OWN = range(W, W + 17)
REL = W + 17; V = REL + 4
ANS = dict(fact=6, sum=3, chain=15, read=11)  # target position of each kind's answer
BASE = np.uint64(1000003); C_EDGES = np.array([1, 2, 4, 8, 16, 64]); P_EDGES = np.array([0.25, 0.5, 0.75, 0.9, 0.9999])


def rows(*cols):
    m = np.full((len(cols[0]), T + 1), PAD, dtype=np.int64)
    for j, c in enumerate(cols):
        m[:, j] = c
    return m


def tell(f, form):
    n = len(f); o = np.full(n, 0); k = lambda t: np.full(n, t); r = f[:, 0] + REL
    x, y, z, u, v = (f[:, i] for i in range(1, 6))
    if form == 1:
        return rows(k(THE), r, k(OF), x, y, z, k(IS), u, v, k(DOT))
    return rows(k(FOR), x, y, z, k(THE), r, k(WAS), u, v, k(DOT))


def facts(rng, n):
    ids = rng.choice(4_000_000, n, replace=False)
    return np.stack([ids % 4, ids // 4 % NAMES + 1, ids // 400 % NAMES + 1, ids // 40000 % NAMES + 1, rng.integers(1, NAMES + 1, n), rng.integers(1, NAMES + 1, n)], 1)


def sums(rng, n, pairs):
    p_ = pairs[rng.integers(0, len(pairs), n)]; k = lambda t: np.full(n, t)
    return rows(p_[:, 0] + DIG, k(PLUS), p_[:, 1] + DIG, k(EQ), (p_[:, 0] + p_[:, 1]) % 10 + DIG, k(DOT))


def chain(rng, n):
    s = rng.random((n, NAMES)).argsort(1)[:, :3] + 1; k = lambda t: np.full(n, t); f = rng.random(n) < 0.5
    a1 = [s[:, 0], k(IS), k(TALLER), k(THAN), s[:, 1], k(DOT)]; a2 = [s[:, 1], k(IS), k(TALLER), k(THAN), s[:, 2], k(DOT)]
    c = [np.where(f, u, v) for u, v in zip(a1, a2)] + [np.where(f, v, u) for u, v in zip(a1, a2)]
    return rows(*c, k(WHO), k(IS), k(TALLEST), k(Q), s[:, 0], k(DOT))


def read(rng, n):
    s = rng.random((n, NAMES)).argsort(1)[:, :4] + 1; k = lambda t: np.full(n, t); f = rng.random(n) < 0.5
    return rows(s[:, 0], k(OWNS), s[:, 1], k(DOT), s[:, 2], k(OWNS), s[:, 3], k(DOT), k(WHAT), k(DOES), np.where(f, s[:, 0], s[:, 2]), k(OWN), np.where(f, s[:, 1], s[:, 3]), k(DOT))


def mix(k):
    k = k ^ (k >> np.uint64(29)); k = k * np.uint64(0xbf58476d1ce4e5b9); k = k ^ (k >> np.uint64(32))
    return k & np.uint64(0xffffffffffff0000)


def kind(count, top):
    return np.searchsorted(C_EDGES, count, side="right") * 6 + np.searchsorted(P_EDGES, top / count, side="left")


def keys(X, n):
    """Hash of the last n tokens at every position; positions with fewer than n tokens get key 0."""
    h = np.zeros(X.shape, dtype=np.uint64); pw = np.uint64(1)
    for i in range(n):
        h[:, n - 1:] += X[:, n - 1 - i:X.shape[1] - i].astype(np.uint64) * pw; pw = pw * BASE
    k = mix(h); k[:, :n - 1] = 0
    return k


def table(L, n):
    X, Y = L[:, :-1], L[:, 1:]; k = keys(X, n); keep = (Y != PAD) & (k != 0)
    a = np.sort(k[keep] | Y[keep].astype(np.uint64))
    new = np.r_[True, a[1:] != a[:-1]]; idx = np.flatnonzero(new); cnt = np.diff(np.r_[idx, len(a)]).astype(np.uint64); u = a[idx]
    uk = u & np.uint64(0xffffffffffff0000); first = np.flatnonzero(np.r_[True, uk[1:] != uk[:-1]])
    score = (cnt << np.uint64(16)) | (u & np.uint64(0xffff))
    top = np.maximum.reduceat(score, first); tot = np.add.reduceat(score >> np.uint64(16), first)
    score[score == np.repeat(top, np.diff(np.r_[first, len(score)]))] = 0
    sec = np.maximum.reduceat(score, first)
    return uk[first], (top & np.uint64(0xffff)).astype(np.int64), (top >> np.uint64(16)).astype(np.int64), (sec & np.uint64(0xffff)).astype(np.int64), (sec >> np.uint64(16)).astype(np.int64), tot.astype(np.int64)


def lookup(t, L, n, loo):
    K, t1, n1, t2, n2, tot = t; X, Y = L[:, :-1], L[:, 1:]; k = keys(X, n)
    i = np.searchsorted(K, k).clip(0, len(K) - 1); hit = (K[i] == k) & (k != 0)
    if loo:
        own = Y; c = tot[i] - 1; a1 = t1[i].copy(); b = n1[i].copy()
        is1 = a1 == own; b = np.where(is1, b - 1, b); use2 = is1 & (n2[i] > b)
        a1 = np.where(use2, t2[i], a1); b = np.where(use2, n2[i], b); hit &= c > 0
    else:
        c = tot[i]; a1 = t1[i]; b = n1[i]
    tok = np.where(hit, a1, -1); kd = np.zeros_like(tok); kd[hit] = kind(c[hit], np.maximum(b[hit], 1))
    return tok.astype(np.int32), kd.astype(np.uint8)


def build():
    t0 = time.time(); rng = np.random.default_rng(0)
    held = np.array([(x, y) for x in range(10) for y in range(10)]); held = held[rng.permutation(100)]; test_pairs, train_pairs = held[:20], held[20:]
    n_rare = int(0.3 * NL) // 3; n_other = int(0.1 * NL) // 3; n_common = 300; n_kind = int(0.15 * NL)
    f = facts(rng, n_rare + n_other + n_common + NTEST)
    rare, other, common, new = f[:n_rare], f[n_rare:n_rare + n_other], f[n_rare + n_other:n_rare + n_other + n_common], f[-NTEST:]
    parts = [tell(np.repeat(rare, 3, 0), 1), tell(np.repeat(other, 3, 0), 2), tell(common[rng.integers(0, n_common, n_kind)], 1),
             sums(rng, n_kind, train_pairs), chain(rng, n_kind), read(rng, n_kind)]
    L = np.concatenate(parts); reserved = np.zeros(len(L), bool); reserved[:6 * NTEST] = True  # the first 2000 rare facts are never trained on
    order = rng.permutation(len(L)); order = order[~reserved[order]]; stream = L[order[:POOL]]
    print("lines", len(L), round(time.time() - t0, 1), flush=True)
    G = {"common_fact": (tell(common[rng.integers(0, n_common, NTEST)], 1), ANS["fact"]), "rare_table": (tell(rare[:NTEST], 1), ANS["fact"]),
         "rare_other_words": (tell(rare[NTEST:2 * NTEST], 2), ANS["fact"]), "fact_not_in_table": (tell(new, 1), ANS["fact"]),
         "held_out_sum": (sums(rng, NTEST, test_pairs), ANS["sum"]), "seen_sum": (sums(rng, NTEST, train_pairs), ANS["sum"]),
         "read_fresh": (read(rng, NTEST), ANS["read"]), "chain_fresh": (chain(rng, NTEST), ANS["chain"])}
    out = dict(stream=stream)
    tt, tk = [], []; gt = {g: [] for g in G}; gk = {g: [] for g in G}
    for n in LEVELS:
        t = table(L, n); s, d = lookup(t, stream, n, True); tt.append(s); tk.append(d)
        for g, (m, _) in G.items():
            s, d = lookup(t, m, n, False); gt[g].append(s); gk[g].append(d)
        print("table", n, len(t[0]), round(time.time() - t0, 1), flush=True)
    out.update(tt=np.stack(tt), tk=np.stack(tk))
    for g, (m, pos) in G.items():
        out.update({f"{g}_x": m, f"{g}_tok": np.stack(gt[g]), f"{g}_kd": np.stack(gk[g]), f"{g}_pos": pos})
    np.savez(CACHE, **out)
    print("built", round(time.time() - t0, 1), flush=True)


if not os.path.exists(CACHE):
    build()
D = np.load(CACHE); used = a.steps * a.batch
stream = torch.as_tensor(D["stream"][:used]); TT = torch.as_tensor(D["tt"][:, :used].astype(np.int64)); TK = torch.as_tensor(D["tk"][:, :used].astype(np.int64))
GROUPS = ["common_fact", "rare_table", "rare_other_words", "fact_not_in_table", "seen_sum", "held_out_sum", "read_fresh", "chain_fresh"]


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
        nk = len(LEVELS) * 42
        s.sigma = nn.Parameter(torch.zeros(nk)); s.beta = nn.Parameter(torch.zeros(nk, a.width)); s.alpha = nn.Parameter(torch.ones(3))

    def hint(s, tok, kd):
        if tok is None:
            return 0.0
        H = 0.0
        for L in USE:
            m = tok[L] >= 0; k = kd[L] + 42 * L
            H = H + (s.sigma[k][..., None] * s.emb(tok[L].clamp(min=0)) + s.beta[k]) * m[..., None]
        return H

    def forward(s, x, tok=None, kd=None):
        H = s.hint(tok, kd); use = lambda site: s.alpha[site] * H if site in SITES else 0.0
        x = s.emb(x) + use(0)
        for i, b in enumerate(s.blocks):
            x = b(x)
            if i == len(s.blocks) // 2 - 1:
                x = x + use(1)
        return (s.norm(x) + use(2)) @ s.emb.weight.T


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
def evaluate(g):
    """Accuracy and loss at the answer token; the table's own accuracy from its longest match."""
    m = torch.as_tensor(D[f"{g}_x"]); pos = int(D[f"{g}_pos"]); x, y = m[:, :-1], m[:, 1:]
    tok = torch.as_tensor(D[f"{g}_tok"].astype(np.int64)); kd = torch.as_tensor(D[f"{g}_kd"].astype(np.int64))
    model.eval()
    out = model(x, tok, kd) if a.hints else model(x)
    off = model(x)
    model.train()
    ans = y[:, pos]; l = out[:, pos]; lo = off[:, pos]
    best = -torch.ones_like(ans)
    for L in range(tok.shape[0]):
        best = tok[L, :, pos].where(tok[L, :, pos] >= 0, best)
    return dict(acc=round((l.argmax(-1) == ans).float().mean().item(), 3), loss=round(F.cross_entropy(l, ans).item(), 3), acc_off=round((lo.argmax(-1) == ans).float().mean().item(), 3),
                table_acc=round((best == ans).float().mean().item(), 3), table_hit=round((best >= 0).float().mean().item(), 3), chance=round(1 / int(ans.unique().numel()), 3))


t0 = time.time()
for step in range(a.steps):
    i = torch.arange(step * a.batch, (step + 1) * a.batch)
    b = stream[i]; x, y = b[:, :-1], b[:, 1:]
    tok, kd = (TT[:, i], TK[:, i]) if a.hints else (None, None)
    if a.hints and a.drop:
        tok = tok.where(torch.rand(tok.shape) > a.drop, torch.tensor(-1))
    loss =F.cross_entropy(model(x, tok, kd).flatten(0, 1), y.flatten(), ignore_index=PAD)
    loss.backward()
    for o in opts:
        for g in o.param_groups:
            g["lr"] = g["base_lr"] * lr_scale(step)
        o.step()
    model.zero_grad(set_to_none=True)
    if not math.isfinite(loss.item()):
        print("not finite"); break
train_s = time.time() - t0
res = dict(vars(a), V=V, params=sum(w.numel() for w in model.parameters()), train_s=round(train_s, 2), last_loss=round(loss.item(), 3), groups={g: evaluate(g) for g in GROUPS})
res["seconds"] = round(time.time() - t0, 2)
print(a.hints, a.lr, a.steps, res["last_loss"], res["seconds"])
for g, r in res["groups"].items():
    print(" ", g, r)
with open(os.path.join(HERE, "facts_results.jsonl"), "a") as f:
    f.write(json.dumps(res) + "\n")
