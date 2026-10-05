"""
plot_fig2.py
════════════
Figure 2: Run B minus Run C differences, drawn from the saved results.

Inputs   results/attribution_similarity.json   (script18)
         results/pubmed_rag_significance.json  (script14)
         discrimination and weight-change values from the script2 log
Outputs  fig_relocation.pdf
"""
import json, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(os.environ.get("RESULTS_DIR", "results"))
OUT = Path(os.environ.get("OUT_DIR", "."))
B_COL, C_COL, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"

att = json.load(open(R / "attribution_similarity.json"))
sig = json.load(open(R / "pubmed_rag_significance.json"))["overall"]


def row(d, lo="ci_lo", hi="ci_hi", m="diff"):
    return d[m], d[lo], d[hi]


panel_a = [("Spearman ρ, signed", row(att["rho_signed"])),
           ("Spearman ρ, |φ|", row(att["rho_abs"])),
           ("Kendall τ, |φ|", row(att["tau"])),
           ("Top-5 Jaccard", row(att["j5"])),
           ("Top-10 Jaccard", row(att["j10"]))]
sg = lambda k: (sig[k]["mean_b"] - sig[k]["mean_c"], sig[k]["ci95_low"], sig[k]["ci95_high"])
panel_b = [("Physiology Jaccard", sg("Physio Jaccard")), ("Physiology RBO", sg("Physio RBO")),
           ("Treatment Jaccard", sg("Treat  Jaccard")), ("Treatment RBO", sg("Treat  RBO"))]
panel_c = [("Attribution-mass shift", sg("Physio |Δattr|")),
           ("Vasopressor AUROC", (0.0169, 0.0106, 0.0235)),     # script2 log, paired bootstrap
           ("Intubation AUROC", (-0.0032, -0.0052, -0.0013))]
weights = {"LSTM\nencoder": ((0.0, 0.0), (0.1650, 0.5158)),       # (mean, max), script2 log
           "Treatment\nMLP": ((0.2118, 0.3229), (0.1429, 0.2850)),
           "Fusion\nhead": ((0.3172, 0.4781), (0.1975, 0.2838))}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))


def forest(ax, rows, title, xlabel, fmt="{:+.3f}", colour=lambda v: B_COL if v >= 0 else C_COL):
    n = len(rows)
    for i, (lab, (m, lo, hi)) in enumerate(rows):
        y = n - 1 - i
        ax.plot([lo, hi], [y, y], color=colour(m), lw=2, solid_capstyle="round")
        ax.plot(m, y, "o", ms=6, color=colour(m), mec="white", mew=1)
        ax.text(1.02, y, f"{fmt.format(m)} [{fmt.format(lo)}, {fmt.format(hi)}]",
                transform=ax.get_yaxis_transform(), va="center", fontsize=7, color=INK)
    ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold", color=INK)
    ax.set_xlabel(xlabel)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


forest(axes[0, 0], panel_a, "(a) Attribution agreement with source, frozen stream",
       "Run B − Run C  (positive: B nearer source)")
forest(axes[0, 1], panel_b, "(b) Retrieved-evidence overlap with source",
       "Run B − Run C  (positive: B nearer source)")
forest(axes[1, 0], panel_c, "(c) Attribution-mass shift and discrimination",
       "Run B − Run C, in each metric's units", fmt="{:+.4f}", colour=lambda v: INK)

ax = axes[1, 1]
names = list(weights)
x = range(len(names))
for j, (lab, col, off) in enumerate((("Run B (bounded)", B_COL, -0.2), ("Run C (unconstrained)", C_COL, 0.2))):
    means = [weights[n][j][0] for n in names]
    maxes = [weights[n][j][1] for n in names]
    ax.bar([i + off for i in x], means, 0.38, color=col, label=lab)
    ax.plot([i + off for i in x], maxes, "_", ms=14, mew=1.5, color=INK)
    for i, v in zip(x, means):
        ax.text(i + off, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=INK)
ax.set_xticks(list(x)); ax.set_xticklabels(names)
ax.set_ylabel("Relative weight change")
ax.set_title("(d) Relative weight change by block", loc="left", fontsize=9, fontweight="bold", color=INK)
ax.set_ylim(0, 0.68)
ax.legend(frameon=False, fontsize=7, loc="upper right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.text(0.01, 0.005, "Bars: 95% paired bootstrap intervals over 300 cases (a–c; AUROC over 5,749 stays). "
         "(d) bar = layer mean, tick = layer maximum; B is seed 2024, C is seed 123.", fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 0.93, 1))
fig.savefig(OUT / "fig_relocation.pdf")
print("wrote", OUT / "fig_relocation.pdf")
