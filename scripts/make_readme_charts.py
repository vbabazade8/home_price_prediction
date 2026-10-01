"""Draws the charts used in README.md and saves them to docs/images/.
Numbers come from notebooks 01, 02 and 04 (see README for details)."""
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path("docs/images")
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#1f6feb"
GREY = "#9aa0a6"
RED = "#d1242f"
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# 1. How the error changed step by step
steps = ["First version\n~950 VIP", "10x more data\n(duplicates in test)",
         "Duplicates\nremoved", "Full catalog\n44k rows", "Tuned\nHistGB"]
mae = [88.6, 42.0, 44.3, 38.2, 37.5]  # thousand AZN

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(steps, mae, color=[GREY, GREY, GREY, GREY, BLUE])
bars[1].set_hatch("//")
bars[1].set_alpha(0.5)
ax.bar_label(bars, labels=[f"{v:.1f}k" for v in mae], padding=3)
ax.set_ylabel("MAE, thousand AZN")
ax.set_title("Prediction error: 88.6k → 37.5k AZN")
ax.text(1, 52, "too optimistic", ha="center", fontsize=9, color="#555")
fig.tight_layout()
fig.savefig(OUT / "mae_history.png")
plt.close(fig)


# 2. Learning curve: does more data help?
rows = [8876, 17752, 26628, 35505]
mae_lc = [48.0, 42.9, 40.2, 38.2]  # thousand AZN, same test set

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(rows, mae_lc, marker="o", color=BLUE)
for r, m in zip(rows, mae_lc):
    ax.annotate(f"{m:.1f}k", (r, m), textcoords="offset points",
                xytext=(0, 8), ha="center")
ax.set_xlabel("Training rows")
ax.set_ylabel("MAE, thousand AZN")
ax.set_ylim(35, 51)
ax.set_title("More data keeps lowering the error")
fig.tight_layout()
fig.savefig(OUT / "learning_curve.png")
plt.close(fig)


# 3. Accuracy vs model size (5-fold CV on the full data)
models = [
    ("ExtraTrees", 482, 36.0),
    ("Random Forest", 318, 37.6),
    ("ExtraTrees (small)", 69, 39.2),
    ("HistGB (manual)", 7.1, 41.4),
    ("HistGB (default)", 0.37, 47.1),
]

fig, ax = plt.subplots(figsize=(8, 4.5))
for name, size, m in models:
    ax.scatter(size, m, s=70, color=GREY if size > 100 else BLUE)
    ax.annotate(name, (size, m), textcoords="offset points",
                xytext=(7, 4), fontsize=9)
for limit, label in [(100, "GitHub file limit"), (512, "Render free RAM")]:
    ax.axvline(limit, color=RED, linestyle="--", linewidth=1)
    ax.text(limit * 1.05, 47.5, label, color=RED, fontsize=9, rotation=90, va="top")
ax.set_xscale("log")
ax.set_xlabel("Model size, MB (log scale)")
ax.set_ylabel("MAE, thousand AZN (lower is better)")
ax.set_ylim(34, 49)
ax.set_title("The most accurate models are too large to deploy")
fig.tight_layout()
fig.savefig(OUT / "accuracy_vs_size.png")
plt.close(fig)

print("saved to", OUT)