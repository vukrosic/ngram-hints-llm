"""Unseen folders, table size, longer context, whole-phrase completion, and cost."""
import glob, json, os, time
import numpy as np
import mlx.core as mx
import domains as d
h = d.h


def phrase(name, tb, w, s, n=96, ctx=64, ahead=5):
    """Greedy next 5 tokens from unseen text: how often are all 5 right?"""
    rng = np.random.default_rng(3); st = rng.permutation(len(s) - ctx - ahead - 1)[:n]; ok = [0, 0]; hits = []
    for on in (0, 1):
        for b in range(0, n, 8):
            X = np.stack([s[i:i + ctx] for i in st[b:b + 8]]); Y = np.stack([s[i + ctx:i + ctx + ahead] for i in st[b:b + 8]])
            for j in range(ahead):
                lo = np.array(d.model(mx.array(X)).astype(mx.float32)[:, -1]); lz = np.logaddexp.reduce(lo, axis=1)
                if on:
                    for r in range(len(X)):
                        ht, hk = h.lookup(tb, X[r]); B = {}
                        for L in range(len(h.LV)):
                            t = ht[L, -1]
                            if t >= 0:
                                B[t] = B.get(t, 0) + w[(hk[L, -1] + h.NK * L) * h.NM + np.searchsorted(h.M_EDGES, np.exp(min(lo[r, t] - lz[r], 0)))]
                        for t, v in B.items():
                            lo[r, t] += v
                X = np.c_[X, lo.argmax(1)]
            ok[on] += int((X[:, -ahead:] == Y).all(1).sum())
    R = dict(name=name + " 5 tokens in a row", n=n, off=ok[0], on=ok[1]); print(json.dumps(R), flush=True)
    open(os.path.join(h.HERE, "domain_results.jsonl"), "a").write(json.dumps(R) + "\n")


def folder_pages(dirs):
    P = []
    for x in dirs:
        P += d.files(x + "/**/*.py")
    return P


# Whole model folders held out, so files of one model never sit on both sides.
D = sorted(x for x in glob.glob(d.SITE + "transformers/models/*") if os.path.isdir(x)); np.random.default_rng(0).shuffle(D)
A, B, C = (folder_pages(x) for x in d.parts(D)); tb, w = d.run("code transformers unseen folders", A, B, C); phrase("transformers unseen folders", tb, w, d.join(C))

P = d.files(d.SITE + "sympy/**/*.py"); A, B, C = d.parts(P)
for f in (.01, .1):
    d.run(f"code sympy {f:.0%} of table", A[:int(len(A) * f)], B, C)
tb, w = d.run("code sympy full table", A, B, C); phrase("sympy", tb, w, d.join(C))
d.run("code sympy context 512", A, B, C, BLOCK=512, BATCH=4, last=120)

# Cost.
s = d.join(C)[:1280]; t0 = time.time()
for i in range(0, 1280, 128):
    h.lookup(tb, s[i:i + 128])
print("lookup ms per token", round((time.time() - t0) / 1280 * 1000, 3), "table MB", round(sum(x.nbytes for t in tb for x in t) / 1e6))
X = mx.array(s[:128][None]); t0 = time.time()
for _ in range(5):
    mx.eval(d.model(X))
print("model ms per token, 128 tokens at once", round((time.time() - t0) / 5 / 128 * 1000, 2))
