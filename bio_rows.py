"""Made-up biographies mixed into FineWeb, with exact and order-free hint tables: python bio_rows.py -> data/bio_rows.npz

400 made-up people, each with a birth city, a university and a company, in four wordings:
  0 " N was born in A."               stored in the table only, for everyone
  1 " The city where N was born is A." and 2 " Where was N born? A."   training text, people 0-199 only
  3 " Which city was N born in? A."   test, never stored or trained
Group "trained" = people 0-199, group "table_only" = people 200-399 (only wording 0, only in the table).
Table corpus = first 25M FineWeb training tokens + the packed training bio chunks + wording 0 of everyone.
Exact keys = last n tokens. Order-free keys = the set of the last n content tokens in the 16 tokens before,
in any order, plus whether the previous token is a content token (content = outside the 500 most frequent tokens).
Training rows get leave-one-out hints, as in web_rows.py.
"""
import os, sys, time
import numpy as np, tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))
ONCE = "--once" in sys.argv  # trained people get one training wording, so their facts are in the table twice like rare web facts
OUT = os.path.join(HERE, "data", "bio_rows_once.npz" if ONCE else "bio_rows.npz")
CORPUS = 25_000_000; L = 65; WIN = 16; TOPK = 500; EOT = 50256; BASE = np.uint64(1000003)
EXACT = [2, 3, 4, 6]; BAG = [2, 3, 4]; NP = 400; NA = 200; WEB_TRAIN = 4000; WEB_VAL = 256; TL = 24
C_EDGES = np.array([1, 2, 4, 8, 16, 64]); P_EDGES = np.array([0.25, 0.5, 0.75, 0.9, 0.9999])
S = np.memmap(os.path.join(HERE, "data", "fineweb_train_000001.bin"), dtype=np.uint16, mode="r", offset=1024)
Wf = np.memmap(os.path.join(HERE, "data", "fineweb_val_000000.bin"), dtype=np.uint16, mode="r", offset=1024)
enc = tiktoken.get_encoding("gpt2"); rng = np.random.default_rng(0); t0 = time.time()

CITY = "Paris London Berlin Tokyo Madrid Rome Chicago Boston Dublin Vienna Prague Oslo Cairo Lima Seoul Delhi Sydney Toronto Denver Atlanta Houston Miami Seattle Phoenix Dallas Austin Moscow Beijing Bangkok Istanbul".split()
UNI = "Harvard Stanford Yale Princeton Oxford Cambridge MIT Columbia Cornell Duke Berkeley Purdue Rice Brown Auburn Clemson Baylor".split()
COMP = "Google Apple Microsoft Amazon Intel IBM Toyota Samsung Sony Nike Disney Boeing Ford Tesla Nokia Oracle Netflix Adobe Cisco Uber Honda Nissan Walmart Visa Shell BMW Audi Dell".split()
TEMPL = {"born": (CITY, [" {n} was born in {a}.", " The city where {n} was born is {a}.", " Where was {n} born? {a}.", " Which city was {n} born in? {a}."]),
         "studied": (UNI, [" {n} studied at {a}.", " The university where {n} studied is {a}.", " Where has {n} studied? {a}.", " Which university has {n} studied at? {a}."]),
         "worked": (COMP, [" {n} worked at {a}.", " The company where {n} worked is {a}.", " Where has {n} worked? {a}.", " Which company has {n} worked at? {a}."])}
SYL = "zar vek mol an tri bel kor ush pen dra lix mu sor ten gal rin fo quel bra nid ost ul ka ry".split()
names = set()
while len(names) < NP:
    s = lambda k: "".join(rng.choice(SYL, k)).capitalize(); names.add(s(2) + " " + s(3))
names = sorted(names); rng.shuffle(names)
facts = [{r: str(rng.choice(TEMPL[r][0])) for r in TEMPL} for _ in range(NP)]


def sent(p, r, w):
    return TEMPL[r][1][w].format(n=names[p], a=facts[p][r])


def prompt(p, r, w):
    t = TEMPL[r][1][w]; return t[:t.index(" {a}")].format(n=names[p]), enc.encode(" " + facts[p][r])[0]


# training bio chunks: wordings 1 and 2 of people 0-199, shuffled and packed into rows of L tokens
bio = [enc.encode(sent(p, r, w)) for p in range(NA) for r in TEMPL for w in (1, 2) if not ONCE or w == 1 + (p + len(r)) % 2]
rows, cur = [], []
for i in rng.permutation(len(bio)):
    if len(cur) + len(bio[i]) > L:
        rows.append(cur + [EOT] * (L - len(cur))); cur = []
    cur += bio[i]
rows.append(cur + [EOT] * (L - len(cur))); BIO = np.array(rows, dtype=np.int64)
store = np.array([t for p in range(NP) for r in TEMPL for t in enc.encode(sent(p, r, 0))], dtype=np.int64)
T = np.concatenate([np.asarray(S[:CORPUS]).astype(np.int64), [EOT], BIO.ravel(), [EOT], store])
stop = np.zeros(65536, bool); stop[np.argsort(np.bincount(T[:CORPUS], minlength=65536))[-TOPK:]] = True

ws = rng.choice(780_000 // L - 1, WEB_TRAIN, replace=False) * L
WEB = np.stack([np.asarray(S[s:s + L]) for s in ws]).astype(np.int64)
vs = rng.choice(1_000_000 // L - 1, WEB_VAL, replace=False) * L
VAL = np.stack([np.asarray(Wf[s:s + L]) for s in vs]).astype(np.int64)
TEST = {}
for g, people, w in [("trained_new_wording", range(NA), 3), ("table_only_new_wording", range(NA, NP), 3), ("table_only_stored_wording", range(NA, NP), 0)]:
    X = np.full((NP // 2 * 3, TL), EOT, dtype=np.int64); pos = np.zeros(len(X), dtype=np.int64); ans = np.zeros(len(X), dtype=np.int64); i = 0
    for p in people:
        for r in TEMPL:
            s, a = prompt(p, r, w); t = enc.encode(s); X[i, :len(t)] = t; pos[i] = len(t) - 1; ans[i] = a; i += 1
    TEST[g] = (X, pos, ans)
print("data", len(T), "bio rows", len(BIO), round(time.time() - t0, 1), flush=True)


def mix(k):
    k = k ^ (k >> np.uint64(29)); k = k * np.uint64(0xbf58476d1ce4e5b9); k = k ^ (k >> np.uint64(32))
    return k & np.uint64(0xffffffffffff0000)


def kind(count, top):
    return np.searchsorted(C_EDGES, count, side="right") * 6 + np.searchsorted(P_EDGES, top / count, side="left")


def keys(T, start, n, bag):
    """Key predicting T[q] from tokens start[q] .. q-1; 0 = no key."""
    q = np.arange(len(T)); out = np.zeros(len(T), dtype=np.uint64)
    if not bag:
        h = np.full(len(T), np.uint64(n)); ok = q - start >= n
        for i in range(1, n + 1):
            h = h * BASE + T[np.clip(q - i, 0, None)].astype(np.uint64)
        return np.where(ok, mix(h), out)
    cpos = np.flatnonzero(~stop[T]); ctok = T[cpos]
    for s in range(0, len(T), 4_000_000):
        qq = q[s:s + 4_000_000]; j = np.searchsorted(cpos, qq, side="left") - 1; ok = (j >= n - 1) & (qq - start[s:s + 4_000_000] >= 1)
        jj = np.clip(j[:, None] - np.arange(n - 1, -1, -1), 0, None); ok &= cpos[jj[:, 0]] >= np.maximum(qq - WIN, start[s:s + 4_000_000])
        h = np.uint64(n + 100) + np.uint64(7) * (~stop[T[np.clip(qq - 1, 0, None)]]).astype(np.uint64)
        for v in np.sort(ctok[jj], 1).T:
            h = h * BASE + v.astype(np.uint64)
        out[s:s + len(qq)] = np.where(ok, mix(h), 0)
    return out


def rkeys(X, n, bag):
    start = np.repeat(np.arange(X.shape[0]) * X.shape[1], X.shape[1])
    return keys(X.ravel(), start, n, bag).reshape(X.shape)


def build(n, bag):
    k = keys(T, np.zeros(len(T), dtype=np.int64), n, bag); a = np.sort((k | T.astype(np.uint64))[k > 0]); del k
    new = np.r_[True, a[1:] != a[:-1]]; idx = np.flatnonzero(new); cnt = np.diff(np.r_[idx, len(a)]).astype(np.uint64); u = a[idx]; del a, new, idx
    uk = u & np.uint64(0xffffffffffff0000); first = np.flatnonzero(np.r_[True, uk[1:] != uk[:-1]])
    score = (cnt << np.uint64(16)) | (u & np.uint64(0xffff)); del cnt, u
    top = np.maximum.reduceat(score, first); tot = np.add.reduceat(score >> np.uint64(16), first)
    score[score == np.repeat(top, np.diff(np.r_[first, len(score)]))] = 0
    sec = np.maximum.reduceat(score, first)
    return (uk[first], (top & np.uint64(0xffff)).astype(np.int64), (top >> np.uint64(16)).astype(np.int64),
            (sec & np.uint64(0xffff)).astype(np.int64), (sec >> np.uint64(16)).astype(np.int64), tot.astype(np.int64))


def lookup(t, k, own=None):
    """Top next token and kind for keys k; with own, that occurrence comes off the counts (leave-one-out)."""
    K, t1, n1, t2, n2, tot = t; i = np.searchsorted(K, k).clip(0, len(K) - 1); hit = (K[i] == k) & (k > 0)
    if own is None:
        c = tot[i]; a1 = t1[i]; b = n1[i]
    else:
        c = tot[i] - 1; a1 = t1[i].copy(); b = n1[i].copy(); is1 = a1 == own; b = np.where(is1, b - 1, b); use2 = is1 & (n2[i] > b)
        a1 = np.where(use2, t2[i], a1); b = np.where(use2, n2[i], b); hit &= c > 0
    tok = np.where(hit, a1, -1); kd = np.zeros_like(tok); kd[hit] = kind(c[hit], np.maximum(b[hit], 1))
    return tok.astype(np.int32), kd.astype(np.uint8)


out = dict(web=WEB, bio=BIO, val=VAL, names=np.array(names), levels=np.array([f"e{n}" for n in EXACT] + [f"b{n}" for n in BAG]))
for g, (X, pos, ans) in TEST.items():
    out.update({f"{g}_x": X, f"{g}_pos": pos, f"{g}_ans": ans})
R = {k_: [] for k_ in ["web_tok", "web_kd", "bio_tok", "bio_kd", "val_tok", "val_kd"] + [f"{g}_{x}" for g in TEST for x in ("tok", "kd")]}
for n, bag in [(n, False) for n in EXACT] + [(n, True) for n in BAG]:
    t = build(n, bag)
    for name, X, loo in [("web", WEB, True), ("bio", BIO, True), ("val", VAL, False)]:
        # row j predicts X[:, j + 1] from X[:, :j + 1]; keys() gives the key predicting position q, so shift by one
        k = rkeys(X, n, bag)[:, 1:]; tok, kd = lookup(t, k.ravel(), X[:, 1:].ravel() if loo else None)
        R[f"{name}_tok"].append(tok.reshape(k.shape)); R[f"{name}_kd"].append(kd.reshape(k.shape))
    for g, (X, pos, ans) in TEST.items():
        k = rkeys(X, n, bag)[:, 1:]; tok, kd = lookup(t, k.ravel()); R[f"{g}_tok"].append(tok.reshape(k.shape)); R[f"{g}_kd"].append(kd.reshape(k.shape))
        r = np.arange(len(X)); a_ = tok.reshape(k.shape)[r, pos]
        print(f"  {'bag' if bag else 'exact'} {n} {g}: hint at answer {np.mean(a_ >= 0):.3f}, right {np.mean(a_ == ans):.3f}", flush=True)
    vt = R["val_tok"][-1]; print(f"{'bag' if bag else 'exact'} {n}: web val share {np.mean(vt >= 0):.3f} right {np.mean(vt == VAL[:, 1:]):.3f}  bio train right {np.mean(R['bio_tok'][-1] == BIO[:, 1:]):.3f}  {round(time.time() - t0, 1)} s", flush=True)
    del t
out.update({k_: np.stack(v) for k_, v in R.items()})
np.savez(OUT, **out)
print("done", round(time.time() - t0, 1))
