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
