"""Figures for the recommender comparison (reads results/recsys/metrics.json).

Usage: .venv/bin/python -m src.reasoning.recsys.plots
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.common.paths import RESULTS  # noqa: E402
from src.perception.camera_trap.plots import T, style  # noqa: E402

OUT = RESULTS / "recsys"
BLUE = "#2a78d6"
LABEL = {
    "popularity_global": "Popularity (global)", "popularity_regional": "Popularity (by country)",
    "knowledge_pure": "Knowledge-based (pure)", "knowledge_popularity": "Knowledge + popularity",
    "collaborative_als": "Collaborative (ALS)", "content_taxonomy": "Content (taxonomy)",
    "content_bioclip": "Content (BioCLIP)",
}
ORDER = list(LABEL)
SEGS = [("cold", "New visitors (no history)"), ("light", "Light users (1–4 species)"), ("heavy", "Active users (5+ species)")]


def by_segment(m: dict) -> None:
    t = m["test"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, (seg, title) in zip(axes, SEGS):
        best = max(ORDER, key=lambda n: t[n][seg]["precision@10"])
        for y, name in enumerate(ORDER):
            r = t[name][seg]
            if r["served"] == 0:
                ax.text(0.003, y, "can't serve: no history", va="center", fontsize=8, color=T["text2"], style="italic")
                continue
            lo, hi = r["precision_ci95"]
            ax.plot([lo, hi], [y, y], color=BLUE, linewidth=2, alpha=0.35, solid_capstyle="round")
            ax.plot(r["precision@10"], y, "o", color=BLUE, markersize=7, markeredgecolor=T["surface"], markeredgewidth=1.2)
            if name == best:
                ax.annotate(f"{r['precision@10']:.3f}", (hi, y), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8.5, color=T["text"], fontweight="bold")
        ax.set_title(f"{title}\nn = {t['popularity_regional'][seg]['users']} test users", loc="left", fontsize=10)
        ax.set_xlim(0, 0.17)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Precision@10")
    axes[0].set_yticks(range(len(ORDER)), [LABEL[n] for n in ORDER])
    axes[0].invert_yaxis()
    fig.suptitle("Which recommender works for whom? Held-out 2026 iNaturalist observers", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, "Dots = precision@10 (share of the top 10 that the user went on to record); lines = 95% bootstrap range. "
             "Target = species new to the user, recorded after 1 Jan 2026.", fontsize=8, color=T["text2"])
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(OUT / "comparison_by_segment.png", dpi=160)
    plt.close(fig)


def accuracy_vs_novelty(m: dict) -> None:
    t = m["test"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    offsets = {"popularity_global": (6, -10), "popularity_regional": (6, 4), "knowledge_pure": (6, -10),
               "knowledge_popularity": (6, 4), "collaborative_als": (6, 4), "content_taxonomy": (-8, -13), "content_bioclip": (6, 4)}
    for name in ORDER:
        r = t[name]["all"]
        ax.plot(r["novelty_bits"], r["hit_rate@10"], "o", color=BLUE, markersize=8, markeredgecolor=T["surface"], markeredgewidth=1.2)
        dx, dy = offsets[name]
        ax.annotate(LABEL[name], (r["novelty_bits"], r["hit_rate@10"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, color=T["text2"], ha="right" if dx < 0 else "left")
    ax.set_xlabel("Novelty of recommendations (bits; higher = less obvious species)")
    ax.set_ylabel("Hit rate@10 (share of users with ≥1 hit)")
    ax.set_ylim(-0.04, 0.85)
    ax.set_xlim(2, 10.5)
    ax.set_title("Accuracy vs novelty: no model wins both", loc="left")
    fig.text(0.01, 0.01, "All 1,869 test users. Collaborative and content models can only serve the 26% of users with a history.",
             fontsize=8, color=T["text2"])
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "accuracy_vs_novelty.png", dpi=160)
    plt.close(fig)


def main() -> None:
    style()
    m = json.loads((OUT / "metrics.json").read_text())
    by_segment(m)
    accuracy_vs_novelty(m)
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
