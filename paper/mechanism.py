"""Draws paper/mechanism.png: how SetGram stores a fact and finds it again when the question uses new words."""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RARE, COMMON, ANS, INK, MUTED, BAD, GOOD = "#2a78d6", "#d9dde3", "#e0741c", "#1f2328", "#5b6470", "#c0392b", "#1e8449"
fig, ax = plt.subplots(figsize=(13, 6.4)); ax.set_xlim(0, 13); ax.set_ylim(0, 6.4); ax.axis("off")


def box(x, y, w, h, fc, ec="none", lw=1.2, r=0.08):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def tokens(x, y, toks, key=()):
    """GPT-2 tokens as chips: blue = rare (outside the 500 most frequent), gray = common, orange = the answer."""
    pos = {}
    for i, (t, kind) in enumerate(toks):
        w = 0.09 * len(t) + 0.18; fc = {"r": RARE, "c": COMMON, "a": ANS}[kind]
        box(x, y, w, 0.42, fc, ec=INK if i in key else "none", lw=2.2 if i in key else 0)
        ax.text(x + w / 2, y + 0.21, t, ha="center", va="center", fontsize=10.5, color="white" if kind != "c" else INK, family="monospace")
        pos[i] = (x, w); x += w + 0.05
    return x


def arrow(x0, y0, x1, y1, c=INK):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6, mutation_scale=14))


ax.text(0.1, 6.1, "SetGram: the key is the set of the last rare words, so the same fact is found when the question is worded differently",
        fontsize=13, weight="bold", color=INK)

# 1. store
ax.text(0.1, 5.45, "1  Store a fact (plain text, no training)", fontsize=11.5, weight="bold", color=INK)
st = [("Zar", "r"), ("ve", "c"), ("k", "r"), ("Mol", "r"), ("ant", "r"), ("ri", "r"), ("was", "c"), ("born", "r"), ("in", "c"), ("Paris", "a")]
e1 = tokens(0.1, 4.75, st, key={4, 5, 7})
ax.text(0.1, 4.45, "last 3 rare words before the next word", fontsize=9.5, color=MUTED)
arrow(e1 + 0.1, 4.96, 5.75, 4.96)
box(5.8, 4.6, 3.0, 0.72, "white", ec=INK)
ax.text(7.3, 5.08, "key = {ant, born, ri}", ha="center", fontsize=11, family="monospace", color=INK)
ax.text(7.3, 4.75, "sorted, so order is lost", ha="center", fontsize=9, color=MUTED)
arrow(8.85, 4.96, 9.55, 4.96)

# table
box(9.6, 3.0, 3.25, 2.3, "#f4f6f8", ec=MUTED)
ax.text(11.22, 5.0, "count table", ha="center", fontsize=11, weight="bold", color=INK)
ax.text(9.75, 4.6, "key", fontsize=9, color=MUTED); ax.text(11.65, 4.6, "next word", fontsize=9, color=MUTED)
for j, (k, v) in enumerate([("{ant, born, ri}", "Paris  1/1"), ("{kor, studied, ush}", "Yale  1/1"), ("{data, model, the}", "set  9/31")]):
    yy = 4.2 - 0.38 * j
    ax.text(9.75, yy, k, fontsize=8.5, family="monospace", color=INK); ax.text(11.65, yy, v, fontsize=8.5, family="monospace", color=ANS if j == 0 else INK)
ax.text(11.22, 3.15, "built from web text + stored facts", ha="center", fontsize=8.5, color=MUTED)

# 2. ask
ax.text(0.1, 3.75, "2  Ask in new words", fontsize=11.5, weight="bold", color=INK)
qt = [("Which", "r"), ("city", "r"), ("was", "c"), ("Zar", "r"), ("ve", "c"), ("k", "r"), ("Mol", "r"), ("ant", "r"), ("ri", "r"), ("born", "r"), ("in", "c"), ("?", "c")]
e2 = tokens(0.1, 3.05, qt, key={7, 8, 9})
arrow(e2 + 0.1, 3.26, 6.75, 3.26)
box(6.8, 2.9, 2.6, 0.72, "white", ec=GOOD, lw=2)
ax.text(8.1, 3.38, "key = {ant, born, ri}", ha="center", fontsize=10.5, family="monospace", color=INK)
ax.text(8.1, 3.05, "same key: found", ha="center", fontsize=9.5, color=GOOD, weight="bold")
arrow(9.45, 3.45, 9.58, 4.15, c=GOOD)

# 3. use
ax.text(0.1, 2.15, "3  Give the model the table's guess", fontsize=11.5, weight="bold", color=INK)
box(0.1, 0.35, 5.6, 1.55, "#f4f6f8", ec=MUTED)
ax.text(0.3, 1.55, "trained with hints:", fontsize=10, weight="bold", color=INK)
ax.text(0.3, 1.18, "embedding of \"Paris\", scaled by a learned trust for", fontsize=9.5, color=INK)
ax.text(0.3, 0.88, "its count and share, is added at the input, the", fontsize=9.5, color=INK)
ax.text(0.3, 0.58, "middle layer and the output", fontsize=9.5, color=INK)
box(5.95, 0.35, 3.5, 1.55, "#f4f6f8", ec=MUTED)
ax.text(6.15, 1.55, "any trained model, no training:", fontsize=10, weight="bold", color=INK)
ax.text(6.15, 1.12, "p = (1 - λs) p_model + λs [Paris]", fontsize=8.8, family="monospace", color=INK)
ax.text(6.15, 0.68, "s = share of \"Paris\" under the key", fontsize=9.5, color=INK)

# comparison: exact keys
box(9.6, 0.35, 3.25, 2.35, "white", ec=BAD, lw=1.6)
ax.text(11.22, 2.38, "exact n-gram key", ha="center", fontsize=11, weight="bold", color=BAD)
ax.text(11.22, 2.08, "(infini-gram, Engram, CoCoNTs)", ha="center", fontsize=8.5, color=MUTED)
ax.text(9.75, 1.68, "stored:", fontsize=9, color=MUTED); ax.text(10.6, 1.68, "\"was born in\"", fontsize=9.5, family="monospace", color=INK)
ax.text(9.75, 1.33, "asked:", fontsize=9, color=MUTED); ax.text(10.6, 1.33, "\"born in ?\"", fontsize=9.5, family="monospace", color=INK)
ax.text(11.22, 0.92, "different key: not found", ha="center", fontsize=9.5, color=BAD, weight="bold")
ax.text(11.22, 0.55, "any change in the last words breaks it", ha="center", fontsize=8.5, color=MUTED)

# legend
for i, (c, t) in enumerate([(RARE, "rare word (outside the 500 most frequent)"), (COMMON, "common word"), (ANS, "answer")]):
    lx = [0.1, 4.0, 5.8][i]; box(lx, 0.0, 0.22, 0.2, c); ax.text(lx + 0.3, 0.1, t, va="center", fontsize=9, color=MUTED)
fig.savefig(os.path.join(HERE, "mechanism.png"), dpi=200, bbox_inches="tight", facecolor="white")
