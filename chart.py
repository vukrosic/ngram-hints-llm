"""Draws facts_vs_reasoning.png from the numbers in RESULTS.md."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASKS = ["Rare fact,\nsame words", "Rare fact,\nother words", "Two-step comparison\n(chance 50%)", "Sums not in the data\n(chance about 10%)"]
ARMS = [("No hints", [.009, .005, 1.0, .492], "#9a9a94"),
        ("Hints at input, middle and output", [1.0, .01, .495, 0], "#e0741c"),
        ("Hints at output only, half dropped", [1.0, .006, .999, .536], "#2a78d6")]
fig, ax = plt.subplots(figsize=(9, 4.6), dpi=160)
w = .25
for i, (name, vals, col) in enumerate(ARMS):
    xs = [x + (i - 1) * w for x in range(len(TASKS))]
    ax.bar(xs, [v * 100 for v in vals], w * .9, color=col, label=name)
ax.set_xticks(range(len(TASKS)), TASKS); ax.set_ylabel("Correct answers (%)"); ax.set_ylim(0, 112)
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e6e6e0"); ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.13), fontsize=9)
fig.tight_layout(); fig.savefig("facts_vs_reasoning.png")

# reworded_recall.png: recall of a fact asked in new words, biographies in web text (bio_run.py, rows from bio_rows.py --once)
METHODS = [("No memory", .067, .038),
           ("Exact n-gram hints (NanoGPT record #367)", .057, .030),
           ("Engram-style hashed 2- and 3-grams", .047, .040),
           ("CoCoNTs-style n-gram targets", .060, .043),
           ("Memory layer with product keys", .062, .055),
           ("Exact table mixed in at test (infini-gram style)", .067, .038),
           ("SetGram hints in training (ours)", .812, .795),
           ("SetGram table mixed in at test (ours)", .963, .958)][::-1]
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
h = .38
for j, (name, col) in enumerate([("People the model trained on", "#2a78d6"), ("People only in the table", "#e0741c")]):
    ys = [i + (0.5 - j) * h for i in range(len(METHODS))]
    vals = [m[1 + j] * 100 for m in METHODS]
    ax.barh(ys, vals, h * .9, color=col, label=name)
    for y, v, m in zip(ys, vals, METHODS):
        if "SetGram" in m[0]:
            ax.text(v + 1, y, f"{v:.0f}%", va="center", fontsize=8, color="#3a3a36")
ax.axvline(4.5, color="#7a7a74", ls="--", lw=1); ax.text(5.5, len(METHODS) - .45, "chance 4.5%", fontsize=8, color="#5a5a55")
ax.set_yticks(range(len(METHODS)), [m[0] for m in METHODS], fontsize=9); ax.set_xlim(0, 105)
ax.set_xlabel("Questions in new words answered right (%)")
ax.set_title("Recall of a fact asked in words never stored or trained on", fontsize=11, loc="left")
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#e6e6e0"); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="center right", fontsize=9)
fig.tight_layout(); fig.savefig("reworded_recall.png")
