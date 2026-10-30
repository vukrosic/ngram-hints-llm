"""Hint rows for web_run.py: python web_rows.py 6  (one n-gram length per call, one file data/rows_6.npz).

Tables over a 100M-token FineWeb corpus are too large to rebuild per run, so this writes only the rows of the fixed
training and validation positions, for each corpus size, once. Corpus sizes are prefixes of the training shard.
Training chunks all lie in the smallest corpus, so every corpus contains the tokens it leaves out.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
n = int(sys.argv[1])
FULL = 100_000_000
SIZES = [FULL >> i for i in range(7, -1, -1)]  # 0.78M ... 100M
L = 65; TRAIN_CHUNKS = 11520; VAL_CHUNKS = 1024; BASE = np.uint64(1000003)
C_EDGES = np.array([1, 2, 4, 8, 16, 64]); P_EDGES = np.array([0.25, 0.5, 0.75, 0.9, 0.9999])
S = np.memmap(os.path.join(HERE, "data", "fineweb_train_000001.bin"), dtype=np.uint16, mode="r", offset=1024)
W = np.memmap(os.path.join(HERE, "data", "fineweb_val_000000.bin"), dtype=np.uint16, mode="r", offset=1024)


def mix(k):
    k = k ^ (k >> np.uint64(29)); k = k * np.uint64(0xbf58476d1ce4e5b9); k = k ^ (k >> np.uint64(32))
    return k & np.uint64(0xffffffffffff0000)  # top 48 bits; the low 16 hold the next token


def kind(count, top):
    return np.searchsorted(C_EDGES, count, side="right") * 6 + np.searchsorted(P_EDGES, top / count, side="left")


def key_at(T, pos):
    h = np.zeros(pos.shape, dtype=np.uint64); pw = np.uint64(1)
    for i in range(1, n + 1):
        h = h + T[pos - i].astype(np.uint64) * pw; pw = pw * BASE
    return mix(h)


rng = np.random.default_rng(0)
tr = rng.permutation(SIZES[0] // L - 1)[:TRAIN_CHUNKS] * L + L  # skips position 0..L so that the longest n-gram fits
tv = rng.permutation(1_000_000 // L - 1)[:VAL_CHUNKS] * L + L
tp = tr[:, None] + np.arange(1, L); vp = tv[:, None] + np.arange(1, L)
kt, kv = key_at(S, tp), key_at(W, vp)

t0 = time.time()
h = np.zeros(FULL, dtype=np.uint64); pw = np.uint64(1)
tok = np.asarray(S[:FULL]).astype(np.uint64)
for i in range(1, n + 1):
    h[i:] += tok[:-i] * pw; pw = pw * BASE
H = mix(h) | tok
del h, tok
print("hashed", round(time.time() - t0, 1), flush=True)

# own-page counts: which next tokens each n-gram has inside each page of the training pool
PC = TRAIN_CHUNKS; EOT = 50256; MASK = 0xffffffffffff0000
pool = SIZES[0] + L
dc = np.cumsum(np.asarray(S[:pool]) == EOT).tolist(); hp = H[:pool].tolist(); pages = {}
for q in range(n, pool):
    e = pages.setdefault((dc[q], hp[q] & MASK), {}); y = hp[q] & 0xffff; e[y] = e.get(y, 0) + 1
kq = kt[:PC].ravel(); dq = np.array(dc)[tp[:PC].ravel()]; del hp


def page_rows(u, cnt):
    """Leave-page-out rows: every match from the token's own page comes off the counts."""
    lo = np.searchsorted(u, kq); hi = np.searchsorted(u, kq | np.uint64(0xffff), side="right")
    pt = -np.ones(kq.shape, dtype=np.int64); total = np.zeros_like(pt); top = np.zeros_like(pt)
    for i in np.flatnonzero(hi > lo):
        c = cnt[lo[i]:hi[i]].astype(np.int64); tk = (u[lo[i]:hi[i]] & np.uint64(0xffff)).astype(np.int64)
        for t, m in pages.get((int(dq[i]), int(kq[i])), {}).items():
            j = np.searchsorted(tk, t)
            if j < len(tk) and tk[j] == t:
                c[j] = max(0, c[j] - m)
        s = c.sum()
        if s > 0:
            j = c.argmax(); pt[i] = tk[j]; total[i] = s; top[i] = c[j]
    pk = np.zeros_like(pt); m = pt >= 0; pk[m] = kind(total[m], top[m])
    return pt.reshape(PC, -1).astype(np.int32), pk.reshape(PC, -1).astype(np.uint8)


loo_tok, loo_kd, val_tok, val_kd, share, page_tok, page_kd = [], [], [], [], [], [], []
for C in SIZES:
    a = np.sort(H[n:C])
    new = np.r_[True, a[1:] != a[:-1]]; idx = np.flatnonzero(new); cnt = np.diff(np.r_[idx, len(a)]).astype(np.uint64); u = a[idx]; del a, new, idx
    ptk, ppk = page_rows(u, cnt); page_tok.append(ptk); page_kd.append(ppk)
    uk = u & np.uint64(0xffffffffffff0000); first = np.flatnonzero(np.r_[True, uk[1:] != uk[:-1]])
    score = (cnt << np.uint64(16)) | (u & np.uint64(0xffff)); del cnt, u
    top = np.maximum.reduceat(score, first); tot = np.add.reduceat(score >> np.uint64(16), first)
    sizes = np.diff(np.r_[first, len(score)])
    score[score == np.repeat(top, sizes)] = 0
    sec = np.maximum.reduceat(score, first); del score, sizes
    K = uk[first]; del uk, first
    t1 = (top & np.uint64(0xffff)).astype(np.int64); n1 = (top >> np.uint64(16)).astype(np.int64)
    t2 = (sec & np.uint64(0xffff)).astype(np.int64); n2 = (sec >> np.uint64(16)).astype(np.int64); tot = tot.astype(np.int64)
    del top, sec

    i = np.searchsorted(K, kv).clip(0, len(K) - 1); hit = K[i] == kv
    vt = np.where(hit, t1[i], -1); vk = np.zeros_like(vt); vk[hit] = kind(tot[i][hit], n1[i][hit])
    val_tok.append(vt.astype(np.int32)); val_kd.append(vk.astype(np.uint8))

    i = np.searchsorted(K, kt).clip(0, len(K) - 1); hit = K[i] == kt  # leave-one-out: the own occurrence comes off the counts
    own = S[tp].astype(np.int64); c = tot[i] - 1; a1 = t1[i].copy(); b = n1[i].copy()
    is1 = a1 == own; b = np.where(is1, b - 1, b); use2 = is1 & (n2[i] > b)
    a1 = np.where(use2, t2[i], a1); b = np.where(use2, n2[i], b); hit &= c > 0
    lt = np.where(hit, a1, -1); lk = np.zeros_like(lt); lk[hit] = kind(c[hit], np.maximum(b[hit], 1))
    loo_tok.append(lt.astype(np.int32)); loo_kd.append(lk.astype(np.uint8))
    share.append((C, round(float((vt >= 0).mean()), 4), round(float((lt >= 0).mean()), 4)))
    print(C, "val share", share[-1][1], "train share", share[-1][2], "page share", round(float((ptk >= 0).mean()), 4), round(time.time() - t0, 1), flush=True)
    del K, t1, n1, t2, n2, tot

# earlier-steps rows: a chunk counts only chunks of earlier steps (8 chunks per step)
seen_tok = -np.ones(kt.shape, dtype=np.int32); seen_kd = np.zeros(kt.shape, dtype=np.uint8); D = {}
for s in range(0, TRAIN_CHUNKS, 8):
    for c in range(s, min(s + 8, TRAIN_CHUNKS)):
        for j, k in enumerate(kt[c].tolist()):
            e = D.get(k)
            if e:
                b = max(e, key=e.get); seen_tok[c, j] = b; seen_kd[c, j] = int(kind(np.array([sum(e.values())]), np.array([e[b]]))[0])
    for c in range(s, min(s + 8, TRAIN_CHUNKS)):
        for k, y in zip(kt[c].tolist(), S[tp[c]].tolist()):
            e = D.setdefault(k, {}); e[y] = e.get(y, 0) + 1
print("seen share", round(float((seen_tok[:1600] >= 0).mean()), 4))

np.savez(os.path.join(HERE, "data", f"rows_{n}.npz"), sizes=np.array(SIZES), x=S[tp - 1].astype(np.int32), y=S[tp].astype(np.int32),
         vx=W[vp - 1].astype(np.int32), vy=W[vp].astype(np.int32), seen_tok=seen_tok, seen_kd=seen_kd,
         loo_tok=np.stack(loo_tok), loo_kd=np.stack(loo_kd), val_tok=np.stack(val_tok), val_kd=np.stack(val_kd),
         page_tok=np.stack(page_tok), page_kd=np.stack(page_kd))
print("done", round(time.time() - t0, 1))
