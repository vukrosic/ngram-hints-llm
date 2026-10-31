"""Five second test: can a learned switch stop hints from blocking reasoning?

python gate.py --arm switch

Half of each batch is facts the table knows and the model has rarely seen. Half is comparison: "a > b . b > c . top ? a ."
with the two statements in random order, tested on name triples held out of training and of the table.
Arms: none, always, both (every step with and without hints), output, switch (learned from the loss),
right (switch trained to predict whether the hint is right), inline (same target, but each switch reads the hidden state at
its own place, so it needs no second pass). Switches set by hand on comparison lines:
noanswer (off at the answer word only), answeronly (on at the answer word only), noline (off on the whole line),
nogt, noname, nodot (off at those words only), nofirstdot (off at the dot between the two statements only).
Hints with the always arm: --late N --freeze M (no hints for N steps, then only trust trains for M steps), --small F (training table reads F of the
lines not trained on, testing reads all, --testsmall 1 keeps it at testing), --names N and --chains N set copies in the table, --levels, --nokind, --dropline, --probe, --cut, --pool, --testseen test the cause, --split 1 (table reads one half of the name triples, training uses the other half), --unique 1 (each triple on one line).
"""
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--arm", default="switch")
p.add_argument("--steps", type=int, default=1000)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--width", type=int, default=32)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--names", type=int, default=20)
p.add_argument("--facts", type=int, default=2000)
p.add_argument("--lr", type=float, default=0.02)
p.add_argument("--hint-lr", type=float, default=0.1)
p.add_argument("--embed-lr", type=float, default=0.003)
p.add_argument("--momentum", type=float, default=0.95)
p.add_argument("--warmup", type=int, default=20)
p.add_argument("--cooldown", type=float, default=0.3)
p.add_argument("--sites", default="", help="where hints enter: 0 input, 1 middle, 2 output")
p.add_argument("--min-share", type=int, default=0, help="keep hints whose top next token has at least this share bucket, 5 = always the same next token")
p.add_argument("--hard", type=int, default=0, help="1 = the switch is fully on or fully off")
p.add_argument("--late", type=int, default=0, help="train this many steps with no hints first")
p.add_argument("--freeze", type=int, default=0, help="then train only the hint trust for this many steps")
p.add_argument("--unique", type=int, default=0, help="1 = each training name triple appears on one line, so hints cannot give away its answer")
p.add_argument("--split", type=int, default=0, help="1 = the hint table reads one half of the comparison triples, training uses the other half, lines still repeat")
p.add_argument("--small", type=float, default=0, help="training hints read this share of the corpus, testing reads all of it")
p.add_argument("--testsmall", type=int, default=0, help="1 = testing also reads the small table")
p.add_argument("--probe", type=int, default=0, help="1 = after training, remove hints one word position at a time")
p.add_argument("--levels", default="", help="keep only these hint lengths, 0 to 3 = lengths 2, 3, 4, 6")
p.add_argument("--dropline", type=float, default=0, help="this share of training comparison lines loses its length 6 hints")
p.add_argument("--measure", type=int, default=0, help="1 = print the share of training lines with no length 6 hint, then stop")
p.add_argument("--nokind", type=int, default=0, help="1 = every hint gets the same trust, no count or share signal")
p.add_argument("--chains", type=int, default=0, help="comparison lines to make, the table reads those not trained on, sets copies of each line")
p.add_argument("--cut", default="", help="no hints at these word positions on comparison lines")
p.add_argument("--pool", type=int, default=0, help="training repeats this many comparison lines")
p.add_argument("--countscale", type=int, default=0, help="1 with --chains = scale test counts by table lines over all lines")
p.add_argument("--testseen", type=int, default=0, help="the table used at testing also holds this many copies of each test line")
p.add_argument("--every", type=int, default=0, help="print comparison this often")
p.add_argument("--two", type=int, default=0, help="1 = every step also trains without hints")
p.add_argument("--tag", default="")
a = p.parse_args()
torch.manual_seed(0); torch.set_num_threads(1); rng = np.random.default_rng(0)
PAD = 0; N = a.names; GT, DOT, TOP, Q, IS = range(N + 1, N + 6); V = N + 6; T = 12; LV = [2, 3, 4, 6]; COPIES = 5; NCHAIN = a.chains or 100_000
FPOS, CPOS = 3, 9; BASE = np.uint64(1000003); C_EDGES = np.array([1, 2, 4, 8, 16, 64]); P_EDGES = np.array([0.25, 0.5, 0.75, 0.9, 0.9999])
SITES = {int(c) for c in a.sites.split(",")} if a.sites else {2} if a.arm == "output" else {1, 2} if a.arm == "inline" else {0, 1, 2}
HAND = ("noanswer", "answeronly", "noline", "nogt", "noname", "nodot", "nofirstdot"); HINTS = a.arm != "none"; TWO = a.two or a.arm in ("both", "switch", "right") + HAND


def lines(*cols):
    m = np.full((len(cols[0]), T + 1), PAD, dtype=np.int64)
    for j, c in enumerate(cols):
        m[:, j] = c
    return m


def chains(s):
    n = len(s); k = lambda t: np.full(n, t); f = rng.random(n) < .5
    one = [s[:, 0], k(GT), s[:, 1], k(DOT)]; two = [s[:, 1], k(GT), s[:, 2], k(DOT)]
    return lines(*[np.where(f, u, v) for u, v in zip(one, two)], *[np.where(f, v, u) for u, v in zip(one, two)], k(TOP), k(Q), s[:, 0], k(DOT))


F_ = rng.integers(1, N + 1, (a.facts, 4)); F_[:, 3] = rng.integers(1, N + 1, a.facts)
F_L = lines(F_[:, 0], F_[:, 1], F_[:, 2], np.full(a.facts, IS), F_[:, 3], np.full(a.facts, DOT))
trip = np.stack(np.meshgrid(*[np.arange(1, N + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
trip = trip[(trip[:, 0] != trip[:, 1]) & (trip[:, 0] != trip[:, 2]) & (trip[:, 1] != trip[:, 2])]; rng.shuffle(trip)
TEST_T, TRAIN_T = trip[:1000], trip[1000:]
C_L = chains(TRAIN_T if a.unique else TRAIN_T[rng.integers(0, len(TRAIN_T), NCHAIN)]); C_TEST = chains(TEST_T)
C_T = C_L
if a.split:
    A_T = TRAIN_T[:len(TRAIN_T) // 2]; C_T = chains(A_T[rng.integers(0, len(A_T), NCHAIN)])
    B_T = TRAIN_T[len(TRAIN_T) // 2:]; C_L = chains(B_T[rng.integers(0, len(B_T), NCHAIN)])
corpus = np.concatenate([np.repeat(F_L, COPIES, 0), C_T])
half = a.batch // 2; need = a.steps * half
stream = np.empty((a.steps * a.batch, T + 1), dtype=np.int64)
stream[::2] = F_L[rng.integers(0, a.facts, need)]; stream[1::2] = C_L[np.arange(need) % (a.pool or len(C_L))]


def mix(k):
    k = k ^ (k >> np.uint64(29)); k = k * np.uint64(0xbf58476d1ce4e5b9); k = k ^ (k >> np.uint64(32))
    return k & np.uint64(0xffffffffffff0000)


def keys(X, m):
    h = np.zeros(X.shape, dtype=np.uint64); pw = 1
    for i in range(m):
        h[:, m - 1:] += X[:, m - 1 - i:X.shape[1] - i].astype(np.uint64) * np.uint64(pw); pw = pw * 1000003 % 2 ** 64
    o = mix(h); o[:, :m - 1] = 0
    return o


def kind(c, t):
    return np.searchsorted(C_EDGES, c, side="right") * 6 + np.searchsorted(P_EDGES, t / c, side="left")


def table(m):
    X, Y = corpus[:, :-1], corpus[:, 1:]; q = keys(X, m); keep = (q != 0) & (Y != PAD)
    u, cnt = np.unique(q[keep] | Y[keep].astype(np.uint64), return_counts=True); uk = u & np.uint64(0xffffffffffff0000)
    first = np.flatnonzero(np.r_[True, uk[1:] != uk[:-1]])
    sc = (cnt.astype(np.uint64) << np.uint64(16)) | (u & np.uint64(0xffff))
    top = np.maximum.reduceat(sc, first); tot = np.add.reduceat(cnt, first)
    sc[sc == np.repeat(top, np.diff(np.r_[first, len(sc)]))] = 0
    sec = np.maximum.reduceat(sc, first)
    return uk[first], (top & np.uint64(0xffff)).astype(np.int64), (top >> np.uint64(16)).astype(np.int64), (sec & np.uint64(0xffff)).astype(np.int64), (sec >> np.uint64(16)).astype(np.int64), tot.astype(np.int64)


def lookup(t, M, m, own):
    K, t1, n1, t2, n2, tot = t; X, Y = M[:, :-1], M[:, 1:]; q = keys(X, m)
    i = np.searchsorted(K, q).clip(0, len(K) - 1); hit = (K[i] == q) & (q != 0) & (Y != PAD)
    if own:
        c = tot[i] - 1; b = n1[i].copy(); h = t1[i].copy(); is1 = h == Y; b = np.where(is1, b - 1, b); use2 = is1 & (n2[i] > b)
        h = np.where(use2, t2[i], h); b = np.where(use2, n2[i], b); hit &= c > 0
    else:
        c = tot[i]; h = t1[i]; b = n1[i]
    tok = np.where(hit, h, -1); kd = np.zeros_like(tok); kd[hit] = kind(np.ceil(c[hit] * CS), np.maximum(np.ceil(b[hit] * CS), 1))
    return tok, kd


def hints(M, own):
    r = [lookup(t, M, m, own) for t, m in zip(TABLES, LV)]
    return torch.as_tensor(np.stack([x[0] for x in r])), torch.as_tensor(np.stack([x[1] for x in r]))


CS = 1.0; TABLES = [table(m) for m in LV]
if a.small:
    full = TABLES; corpus0 = corpus; rows = np.repeat(F_L, COPIES, 0); spare = C_L[need:]
    corpus = np.concatenate([rows[rng.random(len(rows)) < a.small], spare[rng.random(len(spare)) < a.small]]); TABLES = [table(m) for m in LV]
    TT, TK = hints(stream, False)
    if not a.testsmall:
        TABLES = full; CS = a.small * (len(spare) / len(C_T) if a.countscale else 1)
else:
    TT, TK = hints(stream, True)
if a.testseen:
    corpus = np.concatenate([corpus0, np.repeat(C_TEST, a.testseen, 0)]); TABLES = [table(m) for m in LV]
FT, FK = hints(F_L, False); CT, CK = hints(C_TEST, False)
if a.min_share:
    TT, FT, CT = (t.where(k % 6 >= a.min_share, torch.tensor(-1)) for t, k in ((TT, TK), (FT, FK), (CT, CK)))
if a.measure:
    print("no length 6 hint at word 7, all comparison training lines:", round((TT[3, 1::2, 7] < 0).float().mean().item(), 4), "lines", int((TT[3, 1::2, 7] < 0).sum()), flush=True); raise SystemExit
if a.levels:
    for L in set(range(len(LV))) - {int(c) for c in a.levels.split(",")}:
        TT[L] = -1; FT[L] = -1; CT[L] = -1
if a.nokind:
    TK[:] = 0; FK[:] = 0; CK[:] = 0
if a.cut:
    for w in a.cut.split(","):
        TT[:, 1::2, int(w)] = -1; CT[:, :, int(w)] = -1
if a.dropline:
    TT[3, np.flatnonzero((rng.random(len(stream)) < a.dropline) & (np.arange(len(stream)) % 2 == 1))] = -1
S, FL, CL, CT_ABC = torch.as_tensor(stream), torch.as_tensor(F_L), torch.as_tensor(C_TEST), torch.as_tensor(TEST_T)


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
        s.sw = nn.Parameter(torch.zeros(3, a.width)); s.sw0 = nn.Parameter(torch.full((3,), 2.0))

    def hint(s, tok, kd):
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

    def hinted(s, x, H, g):
        s.logs = []

        def use(site, h):
            if site not in SITES:
                return 0.0
            if a.arm != "inline":
                return g * s.alpha[site] * H
            l = s.norm(h.detach()) @ s.sw[site] + s.sw0[site]; s.logs.append(l)
            return s.gate(l) * s.alpha[site] * H

        h = s.emb(x)
        h = h + use(0, h)
        for i, b in enumerate(s.blocks):
            h = b(h)
            if i == len(s.blocks) // 2 - 1:
                h = h + use(1, h)
        return s.norm(h) + use(2, h)

    def gate(s, l):
        return ((l > 0).float() if a.hard else l.sigmoid())[..., None]

    def forward(s, x, tok=None, kd=None):
        """Returns output with hints or None, output without hints or None, and the switch score."""
        if tok is None:
            return None, s.plain(x) @ s.emb.weight.T, None
        H = s.hint(tok, kd); h0 = s.plain(x) if TWO else None
        if a.arm in ("switch", "right"):
            logit = h0.detach() @ s.sw[0] + s.sw0[0]; g = s.gate(logit)
        elif a.arm in HAND:
            logit = None; line = (x == GT).any(1, keepdim=True); ans = x == Q
            on = {"noanswer": ~ans, "answeronly": ans, "noline": torch.zeros_like(ans), "nogt": x != GT, "noname": (x > N) | (x == PAD), "nodot": x != DOT, "nofirstdot": torch.arange(x.shape[1]) != 3}[a.arm]
            g = (~line | on)[..., None].float()
        else:
            logit = None; g = 1.0
        o = s.hinted(x, H, g) @ s.emb.weight.T
        return o, None if h0 is None else h0 @ s.emb.weight.T, s.logs[-1] if a.arm == "inline" else logit


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
hid = [w for n_, w in model.named_parameters() if w.ndim == 2 and "emb" not in n_ and n_ not in ("beta", "sw")]
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


def ce(o, y):
    return F.cross_entropy(o.flatten(0, 1), y.flatten(), ignore_index=PAD)


@torch.no_grad()
def report(M, pos, tok, kd):
    model.eval(); x, y = M[:, :-1], M[:, 1:][:, pos]; off = model(x)[1][:, pos].argmax(-1); r = dict(off=(off == y).float().mean().item())
    longest = tok[:, :, pos].max(0).values
    last = tok[(tok[:, :, pos] >= 0).int().cumsum(0).argmax(0), torch.arange(len(x)), pos]
    r["hint_right"] = (last == y).float().mean().item()
    if HINTS:
        o, _, logit = model(x, tok, kd); on = o[:, pos].argmax(-1)
        r.update(on=(on == y).float().mean().item(), follows=(on == last).float().mean().item())
        if pos == CPOS:
            first = x[:, 0]; hid_ = first == y; r.update(a_hidden=(on == y)[hid_].float().mean().item(), a_shown=(on == y)[~hid_].float().mean().item())
            r.update(picks_first_name=(on == first).float().mean().item(), picks_middle=(on == CT_ABC[:, 1]).float().mean().item(), picks_last=(on == CT_ABC[:, 2]).float().mean().item())
        if logit is not None:
            r["switch"] = logit[:, pos].sigmoid().mean().item()
            if pos == CPOS:
                print("  switch by word:", [round(v, 2) for v in logit.sigmoid().mean(0)[:10].tolist()], flush=True)
    model.train()
    return {k: round(v, 3) for k, v in r.items()}


t0 = time.time()
for step in range(a.steps):
    if a.every and (step % a.every == 0 or step in (a.late, a.late + a.freeze)):
        c_, f_ = report(CL, CPOS, CT, CK), report(FL, FPOS, FT, FK)
        print(f"  step {step} comparison off {c_['off']} on {c_.get('on')} first {c_.get('picks_first_name')}  facts off {f_['off']} on {f_.get('on')}  trust {[round(v, 2) for v in model.alpha.tolist()]}", flush=True)
    i = torch.arange(step * a.batch, (step + 1) * a.batch); b = S[i]; x, y = b[:, :-1], b[:, 1:]
    if HINTS and step >= a.late:
        o, o0, logit = model(x, TT[:, i], TK[:, i]); loss = ce(o, y)
        if o0 is not None:
            loss = (loss + ce(o0, y)) / 2
        if a.arm in ("right", "inline"):
            tok = TT[:, i]; has = (tok >= 0).any(0); last = tok[(tok >= 0).int().cumsum(0).argmax(0), torch.arange(len(x))[:, None], torch.arange(T)]
            for l in (model.logs if a.arm == "inline" else [logit]):
                loss = loss + F.binary_cross_entropy_with_logits(l[has], (last[has] == y[has]).float())
    else:
        loss = ce(model(x)[1], y)
    loss.backward()
    for o_ in opts[1:2] if a.late <= step < a.late + a.freeze else opts:
        for g in o_.param_groups:
            g["lr"] = g["base_lr"] * lr_scale(step)
        o_.step()
    model.zero_grad(set_to_none=True)
    if not math.isfinite(loss.item()):
        print("not finite"); break
if a.every:
    def right(t, y):
        h = t[:, :, CPOS]; r = (h == y[:, CPOS]).float()
        return [round(v, 3) for v in r.mean(1).tolist()], round(r.max(0).values.mean().item(), 3)
    print("  answer hint right by length, any length: train", right(TT[:, 1::2], S[1::2, 1:]), "test", right(CT, CL[:, 1:]), flush=True)
    n = 1000; R = S[1::2][:n]; rt = TT[:, 1::2][:, :n]; rk = TK[:, 1::2][:, :n]; print("  training lines", report(R, CPOS, rt, rk), flush=True)
if a.probe:
    @torch.no_grad()
    def acc(M, tok, kd):
        model.eval()
        return round((model(M[:, :-1], tok, kd)[0][:, CPOS].argmax(-1) == M[:, 1:][:, CPOS]).float().mean().item(), 3)

    for name, M, t, k in (("train", S[1::2][:1000], TT[:, 1::2][:, :1000], TK[:, 1::2][:, :1000]), ("test", CL, CT, CK)):
        drop, only = [], []
        for w in range(CPOS + 1):
            d = t.clone(); d[:, :, w] = -1; drop.append(acc(M, d, k))
            o = torch.full_like(t, -1); o[:, :, w] = t[:, :, w]; only.append(acc(M, o, k))
        print(f"  {name} all {acc(M, t, k)} remove one position {drop} keep one position {only}", flush=True)
s_ = round(time.time() - t0, 1)
res = dict(vars(a), fact=report(FL, FPOS, FT, FK), comp=report(CL, CPOS, CT, CK), train=report(S[1::2][:1000], CPOS, TT[:, 1::2][:, :1000], TK[:, 1::2][:, :1000]), seconds=s_, last_loss=round(loss.item(), 3))
print(f"{a.arm:8} facts {res['fact']}  comparison {res['comp']}  {s_}s", flush=True)
with open(os.path.join(HERE, "gate_results.jsonl"), "a") as f:
    f.write(json.dumps(res) + "\n")
