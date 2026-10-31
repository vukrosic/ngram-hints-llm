"""Order-free hint rows for web_run.py: python web_bag_rows.py 2  (writes data/rows_bag2.npz).

Key at a position = the previous token exactly + the set of the last n content tokens in the 16 tokens before it,
in any order. Content tokens = every token outside the 500 most frequent of the corpus. Corpus = the first 25M
training tokens (corpus index 5 of web_rows.py); other corpus indices hold no hints. Same training and validation
positions and the same leave-one-out as web_rows.py.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
n = int(sys.argv[1]); WIN = 16; TOPK = 500
FULL = 100_000_000; SIZES = [FULL >> i for i in range(7, -1, -1)]; SI = 5; C = SIZES[SI]
L = 65; TRAIN_CHUNKS = 11520; VAL_CHUNKS = 1024; BASE = np.uint64(1000003)
C_EDGES = np.array([1, 2, 4, 8, 16, 64]); P_EDGES = np.array([0.25, 0.5, 0.75, 0.9, 0.9999])
S = np.memmap(os.path.join(HERE, "data", "fineweb_train_000001.bin"), dtype=np.uint16, mode="r", offset=1024)
W = np.memmap(os.path.join(HERE, "data", "fineweb_val_000000.bin"), dtype=np.uint16, mode="r", offset=1024)


def mix(k):
    k = k ^ (k >> np.uint64(29)); k = k * np.uint64(0xbf58476d1ce4e5b9); k = k ^ (k >> np.uint64(32))
    return k & np.uint64(0xffffffffffff0000)


def kind(count, top):
    return np.searchsorted(C_EDGES, count, side="right") * 6 + np.searchsorted(P_EDGES, top / count, side="left")


t0 = time.time(); T = np.asarray(S[:C]).astype(np.int64)
stop = np.zeros(65536, bool); stop[np.argsort(np.bincount(T, minlength=65536))[-TOPK:]] = True


def bag_keys(T, pos):
    """Key predicting T[pos] for every position in pos (sorted); 0 = no key."""
    cpos = np.flatnonzero(~stop[T]); ctok = T[cpos]; out = np.zeros(len(pos), dtype=np.uint64)
    for s in range(0, len(pos), 4_000_000):
        q = pos[s:s + 4_000_000]; j = np.searchsorted(cpos, q, side="left") - 1; ok = (j >= n - 1) & (q >= 1)
        jj = np.clip(j[:, None] - np.arange(n - 1, -1, -1), 0, None); ok &= cpos[jj[:, 0]] >= q - WIN
        h = (T[q - 1].astype(np.uint64) + np.uint64(n)) * BASE
        for v in np.sort(ctok[jj], 1).T:
            h = h * BASE + v.astype(np.uint64)
        out[s:s + len(q)] = np.where(ok, mix(h), 0)
    return out


rng = np.random.default_rng(0)  # the same positions as web_rows.py
tr = rng.permutation(SIZES[0] // L - 1)[:TRAIN_CHUNKS] * L + L
tv = rng.permutation(1_000_000 // L - 1)[:VAL_CHUNKS] * L + L
tp = tr[:, None] + np.arange(1, L); vp = tv[:, None] + np.arange(1, L)
k_all = bag_keys(T, np.arange(C)); kt = k_all[tp.ravel()]
Wv = np.asarray(W[:1_000_100]).astype(np.int64); kv = bag_keys(Wv, vp.ravel())
print("keys", round(time.time() - t0, 1), "share with key", round(float((k_all > 0).mean()), 3), flush=True)

a = np.sort((k_all | T.astype(np.uint64))[k_all > 0]); del k_all
new = np.r_[True, a[1:] != a[:-1]]; idx = np.flatnonzero(new); cnt = np.diff(np.r_[idx, len(a)]).astype(np.uint64); u = a[idx]; del a, new, idx
uk = u & np.uint64(0xffffffffffff0000); first = np.flatnonzero(np.r_[True, uk[1:] != uk[:-1]])
score = (cnt << np.uint64(16)) | (u & np.uint64(0xffff)); del cnt, u
top = np.maximum.reduceat(score, first); tot = np.add.reduceat(score >> np.uint64(16), first)
score[score == np.repeat(top, np.diff(np.r_[first, len(score)]))] = 0
sec = np.maximum.reduceat(score, first); del score
K = uk[first]; del uk, first
t1 = (top & np.uint64(0xffff)).astype(np.int64); n1 = (top >> np.uint64(16)).astype(np.int64)
t2 = (sec & np.uint64(0xffff)).astype(np.int64); n2 = (sec >> np.uint64(16)).astype(np.int64); tot = tot.astype(np.int64)

i = np.searchsorted(K, kv).clip(0, len(K) - 1); hit = (K[i] == kv) & (kv > 0)
vt = np.where(hit, t1[i], -1); vk = np.zeros_like(vt); vk[hit] = kind(tot[i][hit], n1[i][hit])
i = np.searchsorted(K, kt).clip(0, len(K) - 1); hit = (K[i] == kt) & (kt > 0)  # leave-one-out
own = T[tp.ravel()]; c = tot[i] - 1; a1 = t1[i].copy(); b = n1[i].copy()
is1 = a1 == own; b = np.where(is1, b - 1, b); use2 = is1 & (n2[i] > b)
a1 = np.where(use2, t2[i], a1); b = np.where(use2, n2[i], b); hit &= c > 0
lt = np.where(hit, a1, -1); lk = np.zeros_like(lt); lk[hit] = kind(c[hit], np.maximum(b[hit], 1))
print("val share", round(float((vt >= 0).mean()), 4), "train share", round(float((lt >= 0).mean()), 4),
      "val right", round(float((vt == Wv[vp.ravel()]).mean()), 4), "train right", round(float((lt == own).mean()), 4), round(time.time() - t0, 1), flush=True)


def at(x, dt):
    o = (-np.ones((8,) + x.shape, dtype=dt) if dt == np.int32 else np.zeros((8,) + x.shape, dtype=dt)); o[SI] = x
    return o


sh = (TRAIN_CHUNKS, L - 1)
np.savez(os.path.join(HERE, "data", f"rows_bag{n}.npz"), sizes=np.array(SIZES), x=S[tp - 1].astype(np.int32), y=S[tp].astype(np.int32),
         vx=W[vp - 1].astype(np.int32), vy=W[vp].astype(np.int32), seen_tok=-np.ones(sh, np.int32), seen_kd=np.zeros(sh, np.uint8),
         loo_tok=at(lt.reshape(sh).astype(np.int32), np.int32), loo_kd=at(lk.reshape(sh).astype(np.uint8), np.uint8),
         val_tok=at(vt.reshape(VAL_CHUNKS, -1).astype(np.int32), np.int32), val_kd=at(vk.reshape(VAL_CHUNKS, -1).astype(np.uint8), np.uint8),
         page_tok=-np.ones((8,) + sh, np.int32), page_kd=np.zeros((8,) + sh, np.uint8))
print("done", round(time.time() - t0, 1))
