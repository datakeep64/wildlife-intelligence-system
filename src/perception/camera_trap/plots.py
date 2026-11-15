"""Figures for the camera-trap experiment (reads results/camera_trap/*).

Colour roles (reference palette, light mode):
  categorical slots 1-3 -> zero-shot / logistic regression / kNN (validated all-pairs)
  single-hue blue ramp  -> confusion-matrix magnitude
  look-alike embedding panels highlight <= 3 classes each against gray context

Usage: .venv/bin/python -m src.perception.camera_trap.plots
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.common.paths import RESULTS, SPECIES_CSV  # noqa: E402

OUT = RESULTS / "camera_trap"
T = {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e6e5e1", "context": "#d3d2cc"}
SERIES = {"zero_shot_taxonomic": "#2a78d6", "logistic_regression": "#eb6834", "knn": "#1baf7a"}
NAMES = {"zero_shot_taxonomic": "Zero-shot (text prompts)", "logistic_regression": "Logistic regression probe", "knn": "kNN probe"}
SEQ = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": T["surface"], "axes.facecolor": T["surface"], "savefig.facecolor": T["surface"],
        "axes.edgecolor": T["grid"], "axes.labelcolor": T["text2"], "text.color": T["text"],
        "xtick.color": T["text2"], "ytick.color": T["text2"], "axes.grid": True, "grid.color": T["grid"],
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    })


def pretty(keys) -> list[str]:
    common = pd.read_csv(SPECIES_CSV, dtype=str).set_index("key")["common_en"]
    short = {"Southern pig-tailed macaque": "Pig-tailed macaque", "Southern red muntjac": "Red muntjac"}
    return [short.get(common.get(k, k), common.get(k, k)) for k in keys]


def few_shot(m: dict) -> None:
    fs = pd.read_csv(OUT / "few_shot_curve.csv")
    n_all = int(np.mean([v["train"] for v in m["data"]["per_class"].values()]))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for model in ("logistic_regression", "knn"):
        g = fs[fs.model == model].groupby("shots").macro_f1.agg(["mean", "std"])
        x = list(g.index) + [n_all]
        mean = list(g["mean"]) + [m["results"][model]["macro_f1"]]
        std = list(g["std"]) + [0.0]
        c = SERIES[model]
        ax.fill_between(x, np.array(mean) - std, np.array(mean) + std, color=c, alpha=0.14, linewidth=0)
        ax.plot(x, mean, color=c, linewidth=2, marker="o", markersize=5.5,
                markeredgecolor=T["surface"], markeredgewidth=1.2, label=NAMES[model])
        dy = 9 if model == "logistic_regression" else -9  # both end near the same value: split the labels
        ax.annotate(f"{NAMES[model].replace(' probe', '')} {mean[-1]:.2f}", (x[-1], mean[-1]), xytext=(8, dy),
                    textcoords="offset points", va="center", color=T["text2"], fontsize=9)
    zs = m["results"]["zero_shot_taxonomic"]["macro_f1"]
    ax.axhline(zs, color=SERIES["zero_shot_taxonomic"], linewidth=2, linestyle=(0, (5, 3)), label=NAMES["zero_shot_taxonomic"])
    ax.annotate(f"Zero-shot {zs:.2f}", (1, zs), xytext=(2, 6), textcoords="offset points", color=T["text2"], fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(FEW := [1, 2, 5, 10, 20, 50, n_all])
    ax.set_xticklabels([str(v) for v in FEW[:-1]] + [f"all (~{n_all})"])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_xlim(0.8, n_all * 3.2)
    ax.set_xlabel("Labelled training photos per species")
    ax.set_ylabel("Macro-F1 on held-out camera locations")
    ax.set_title("How many labelled photos does it take to beat zero-shot BioCLIP?", loc="left")
    ax.legend(loc="lower right", fontsize=9)
    fig.text(0.01, 0.01, "Bands: ±1 s.d. over 10 random draws of the training photos. Same frozen BioCLIP features for every model.",
             fontsize=8, color=T["text2"])
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "few_shot_curve.png", dpi=160)
    plt.close(fig)


def confusion(m: dict) -> None:
    labels = m["results"]["logistic_regression"]["labels"]
    names = pretty(labels)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for ax, key in zip(axes, ("zero_shot_taxonomic", "logistic_regression")):
        cm = np.array(m["results"][key]["confusion_matrix"], dtype=float)
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(cmn, cmap=SEQ, vmin=0, vmax=1)
        ax.grid(False)
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = cmn[i, j]
                if v >= 0.05:
                    ax.text(j, i, f"{v:.0%}" if v < 0.995 else "100%", ha="center", va="center", fontsize=7.5,
                            color="#ffffff" if v > 0.55 else T["text"])
        ax.set_xticks(range(len(labels)), names, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(labels)), names, fontsize=8.5)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True species")
        r = m["results"][key]
        ax.set_title(f"{NAMES[key]}\naccuracy {r['accuracy']:.2f} · macro-F1 {r['macro_f1']:.2f}", loc="left", fontsize=10)
    fig.text(0.01, 0.01, "Rows sum to 100% (share of each true species). Cells under 5% are left blank. Test set = camera locations never seen in training.",
             fontsize=8, color=T["text2"])
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "confusion_zero_shot_vs_probe.png", dpi=160)
    plt.close(fig)


def per_class(m: dict) -> None:
    labels = m["results"]["logistic_regression"]["labels"]
    order = sorted(labels, key=lambda l: m["results"]["logistic_regression"]["per_class"][l]["recall"])
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    h = 0.26
    for i, key in enumerate(("zero_shot_taxonomic", "logistic_regression", "knn")):
        vals = [m["results"][key]["per_class"][l]["recall"] for l in order]
        ax.barh(np.arange(len(order)) + (1 - i) * (h + 0.02), vals, height=h, color=SERIES[key], label=NAMES[key],
                edgecolor=T["surface"], linewidth=1)
    ax.set_yticks(range(len(order)), pretty(order))
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Recall (share of the species' test photos identified correctly)")
    ax.set_title("Per-species recall on held-out camera locations", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.11), ncol=3, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "per_class_recall.png", dpi=160)
    plt.close(fig)


def embedding(m: dict) -> None:
    u = pd.read_csv(OUT / "umap_coords.csv")
    groups = [("Big cats", ["tiger", "leopard", "clouded_leopard"]),
              ("Sun bear vs wild boar", ["sun_bear", "wild_boar"]),
              ("Deer", ["sambar", "red_muntjac"])]
    colors = list(SERIES.values())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (title, keys) in zip(axes, groups):
        ctx = u[~u.label.isin(keys)]
        ax.scatter(ctx.x, ctx.y, s=6, color=T["context"], linewidths=0, label="Other species")
        for c, k in zip(colors, keys):
            d = u[u.label == k]
            ax.scatter(d.x, d.y, s=16, color=c, edgecolors=T["surface"], linewidths=0.6, label=pretty([k])[0])
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, loc="left")
        ax.legend(loc="best", fontsize=8, markerscale=1.2)
    e = m["embedding"]
    fig.text(0.01, 0.01, f"UMAP of frozen BioCLIP crop embeddings (cosine), train + test photos. "
             f"Silhouette (cosine, 9 species): crop {e['silhouette_cosine_crop']:.3f}, full image {e['silhouette_cosine_full_image']:.3f}.",
             fontsize=8, color=T["text2"])
    fig.suptitle("Where BioCLIP places look-alike species", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(OUT / "embedding_umap.png", dpi=160)
    plt.close(fig)


def main() -> None:
    style()
    m = json.loads((OUT / "metrics.json").read_text())
    few_shot(m); confusion(m); per_class(m); embedding(m)
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
