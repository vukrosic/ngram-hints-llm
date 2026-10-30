"""Splits the web text gain of hints by how rare the text is. Run web_run.py twice with --dump none and --dump all first."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LV = [2, 3, 4, 5, 6, 10, 20]; SIZE = 7
n0 = np.load(os.path.join(HERE, "data", "dump_none.npy")); n1 = np.load(os.path.join(HERE, "data", "dump_all.npy")); gain = n0 - n1
nv = len(gain); R = [np.load(os.path.join(HERE, "data", f"rows_{n}.npz")) for n in LV]
tok = np.stack([r["val_tok"][SIZE][:nv] for r in R]); kd = np.stack([r["val_kd"][SIZE][:nv] for r in R]); y = R[0]["vy"][:nv]
has = tok >= 0; level = np.where(has.any(0), len(LV) - 1 - has[::-1].argmax(0), -1)
i, j = np.indices(level.shape); count = np.where(level >= 0, kd[level.clip(0), i, j] // 6, 0); right = (level >= 0) & (tok[level.clip(0), i, j] == y)
S = np.memmap(os.path.join(HERE, "data", "fineweb_train_000001.bin"), dtype=np.uint16, mode="r", offset=1024)
freq = np.bincount(np.asarray(S[:100_000_000]), minlength=50257)[y]


def show(name, groups):
    print(name, "| share of tokens | share of gain | gain per token | loss without hints | hint right")
    for label, m in groups:
        if m.sum():
            print(f"{label:>22} | {m.mean():.3f} | {gain[m].sum() / gain.sum():.3f} | {gain[m].mean():.3f} | {n0[m].mean():.2f} | {right[m].mean():.2f}")


print("gain", round(float(gain.mean()), 4), "tokens", gain.size)
show("longest match", [("none", level < 0)] + [(str(n), level == k) for k, n in enumerate(LV)])
names = ["no match", "1", "2-3", "4-7", "8-15", "16-63", "64+"]
show("times the match was seen", [(names[c], count == c) for c in range(7)])
edges = [0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e9]
show("times the true next token appears", [(f"{edges[k]:.0f}-{edges[k + 1]:.0f}", (freq >= edges[k]) & (freq < edges[k + 1])) for k in range(6)])
show("hint", [("right", right), ("wrong", (level >= 0) & ~right)])
