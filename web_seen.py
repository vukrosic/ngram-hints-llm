"""Splits the web text gain by how many times the model itself saw the text in its own training, not in the hint table.

Run web_run.py twice with --dump none and --dump all first.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LV = [2, 3, 4, 5, 6, 10, 20]; SIZE = 7; USED = 11520; BASE = np.uint64(1000003)
n0 = np.load(os.path.join(HERE, "data", "dump_none.npy")); n1 = np.load(os.path.join(HERE, "data", "dump_all.npy")); gain = n0 - n1
nv = len(gain); R = [np.load(os.path.join(HERE, "data", f"rows_{n}.npz")) for n in LV]
X, Y, VX, VY = R[0]["x"][:USED], R[0]["y"][:USED], R[0]["vx"][:nv], R[0]["vy"][:nv]
tok = np.stack([r["val_tok"][SIZE][:nv] for r in R]); has = tok >= 0
level = np.where(has.any(0), len(LV) - 1 - has[::-1].argmax(0), -1)
right = (level >= 0) & (np.take_along_axis(tok, level.clip(0)[None], 0)[0] == VY)


def keys(T, m):
    h = np.zeros(T.shape, dtype=np.uint64); pw = np.uint64(1)
    for i in range(m):
        h[:, m - 1:] += T[:, m - 1 - i:T.shape[1] - i].astype(np.uint64) * pw; pw = pw * BASE
    h[:, :m - 1] = 0
    return h


ctx, ans = np.zeros(gain.shape, dtype=np.int64), np.zeros(gain.shape, dtype=np.int64)
for k, m in enumerate(LV):
    tr, va = keys(X, m), keys(VX, m); on = (level == k) & (va != 0)
    for out, a, b in ((ctx, tr, va), (ans, tr * BASE + Y.astype(np.uint64), va * BASE + VY.astype(np.uint64))):
        u, c = np.unique(a[tr != 0], return_counts=True); i = np.searchsorted(u, b[on]).clip(0, len(u) - 1)
        out[on] = np.where(u[i] == b[on], c[i], 0)
ok = (level >= 0) & (np.arange(VX.shape[1])[None] >= np.array(LV)[level.clip(0)] - 1)


def show(name, values):
    print(name, "| share of tokens | share of gain | gain per token | loss without hints | hint right")
    for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 16), (16, 64), (64, 10 ** 9)):
        m = ok & (values >= lo) & (values < hi)
        if m.sum():
            print(f"{lo:>6}-{hi - 1:<10} | {m.mean():.3f} | {gain[m].sum() / gain.sum():.3f} | {gain[m].mean():.3f} | {n0[m].mean():.2f} | {right[m].mean():.2f}")


print("tokens used", round(float(ok.mean()), 3), "gain", round(float(gain.mean()), 4), "model trained on", X.size, "tokens")
show("times the model saw the context", ctx)
show("times the model saw the context with the true next token", ans)
