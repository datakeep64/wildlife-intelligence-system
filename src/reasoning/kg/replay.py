"""Error replay: how do classifier mistakes become alert mistakes?

For every camera-trap TEST photo (334, held-out locations) and each classifier:
  1. the photo becomes an AnimalSighting in Taman Negara with the classifier's species + confidence
  2. a person is detected in the same zone 1 hour later
  3. the SPARQL rules decide the person alert's priority (and whether the sighting goes to review)
  4. compare with the decision the rules make for the TRUE species at full confidence
Camera evidence only (no GBIF zone records), so the outcome depends on the sighting alone.

The rules are deterministic given (predicted taxon, confidence above/below threshold),
so each distinct input is run through the real rule engine once and reused (memoised).

Outputs: results/kg/alert_replay.json, alert_replay_rows.csv
Usage:   .venv/bin/python -m src.reasoning.kg.replay
"""
import itertools
import json
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from src.common.paths import LILA_DIR, RESULTS
from src.perception.camera_trap.classify import KNNProbe, LinearProbe, zero_shot
from src.perception.camera_trap.dataset import load_config
from src.perception.camera_trap.experiment import CACHE, slug
from src.reasoning.kg import rules
from src.reasoning.kg.graph import working_copy
from src.reasoning.kg.sightings import add_event

OUT = RESULTS / "kg"
ZONE = "taman_negara"
T0 = datetime(2026, 11, 10, 2, 0)
THRESHOLDS = [0.0, 0.5, 0.7, 0.9]


@lru_cache(maxsize=None)
def decide(taxon: str, runner_up: str | None, above_threshold: bool, threshold: float,
           review_runner_up: bool) -> tuple[str, bool]:
    """Run the real rule engine once for this input. Returns (person-alert priority, sent to review).
    The runner-up only matters below the threshold, so callers pass None above it (fewer distinct runs)."""
    g = working_copy(include_observations=False)
    conf = min(threshold + 0.05, 0.999) if above_threshold else max(threshold - 0.05, 0.0)
    add_event(g, "s", "animal", ZONE, T0, taxon, conf, runner_up=runner_up)
    add_event(g, "p", "person", ZONE, T0 + timedelta(hours=1))
    rules.run(g, review_threshold=threshold, review_runner_up=review_runner_up)
    review = any(r["sighting"] == "event_s" for r in rules.review_tasks(g))
    return rules.top_priority(g, "p"), review


def classifier_outputs() -> tuple[pd.DataFrame, list[str]]:
    cfg = load_config()
    df = pd.read_csv(LILA_DIR / "manifest.csv", dtype={"location": str, "image_id": str})
    z = np.load(CACHE / f"embeddings_{slug(cfg['model'])}.npz", allow_pickle=True)
    assert z["ids"].tolist() == df.image_id.astype(str).tolist(), "embedding cache out of date: rerun experiment"
    labels = [c["key"] for c in cfg["classes"] if c["key"] in set(df.label)]
    m = json.loads((RESULTS / "camera_trap" / "metrics.json").read_text())["hyperparameters"]
    X, y = z["crop"], df.label.to_numpy()
    tr = ((df.role == "experiment") & (df.split == "train")).to_numpy()
    te = ((df.role == "experiment") & (df.split == "test")).to_numpy()
    _, p_zs = zero_shot(X[te], z["text_taxonomic9"], labels)
    lr = LinearProbe(C=m["logreg_C"]).fit(X[tr], y[tr])
    knn = KNNProbe(k=m["knn_k"]).fit(X[tr], y[tr])
    out = pd.DataFrame({"image_id": df.image_id[te].to_numpy(), "true": y[te]})
    for name, probs, classes in [("zero_shot", p_zs, labels),
                                 ("logistic_regression", lr.predict_proba(X[te]), list(lr.clf.classes_)),
                                 ("knn", knn.clf.predict_proba(X[te]), list(knn.clf.classes_))]:
        order = np.argsort(-probs, axis=1)
        out[f"{name}_pred"] = np.array(classes)[order[:, 0]]
        out[f"{name}_conf"] = probs.max(1)
        out[f"{name}_runner"] = np.array(classes)[order[:, 1]]
    return out, labels


def main() -> None:
    preds, labels = classifier_outputs()
    truth = {lab: decide(lab, None, True, 0.0, True)[0] for lab in labels}
    print("true-label decision per species:", truth)
    rows = []
    summary = {"zone": ZONE, "true_priority_by_species": truth,
               "results": {}, "results_before_fix": {}}  # after / before rule C4 (runner-up review)
    for fix, clf, tau in itertools.product((False, True), ("zero_shot", "logistic_regression", "knn"), THRESHOLDS):
        if True:
            r = preds[["image_id", "true"]].copy()
            r["pred"], r["conf"], r["runner"] = preds[f"{clf}_pred"], preds[f"{clf}_conf"], preds[f"{clf}_runner"]
            dec = [decide(p, None if c >= tau else ru, c >= tau, tau, fix) for p, c, ru in zip(r.pred, r.conf, r.runner)]
            r["priority"] = [d[0] for d in dec]
            r["review"] = [d[1] for d in dec]
            r["true_priority"] = r.true.map(truth)
            r["classifier"], r["threshold"], r["runner_up_rule"] = clf, tau, fix
            rows.append(r)
            hi_true, hi_pred = r.true_priority == "HIGH", r.priority == "HIGH"
            wrong = r.pred != r.true
            summary["results" if fix else "results_before_fix"][f"{clf}@{tau}"] = {
                "classifier": clf, "threshold": tau, "n": int(len(r)),
                "species_errors": int(wrong.sum()),
                "priority_correct": round(float((r.priority == r.true_priority).mean()), 4),
                "true_high": int(hi_true.sum()),
                "high_recall": round(float((hi_true & hi_pred).sum() / hi_true.sum()), 4),
                "high_precision": round(float((hi_true & hi_pred).sum() / max(hi_pred.sum(), 1)), 4),
                "false_high": int((~hi_true & hi_pred).sum()),
                "false_high_examples": r[~hi_true & hi_pred].true.value_counts().to_dict(),
                "missed_high": int((hi_true & ~hi_pred).sum()),
                "missed_high_sent_to_review": int((hi_true & ~hi_pred & r.review).sum()),
                "missed_high_silently": int((hi_true & ~hi_pred & ~r.review).sum()),
                "review_rate": round(float(r.review.mean()), 4),
                "review_precision": round(float((r.review & wrong).sum() / max(r.review.sum(), 1)), 4),
                "errors_caught_by_review": round(float((r.review & wrong).sum() / max(wrong.sum(), 1)), 4),
                "silent_priority_errors": int(((r.priority != r.true_priority) & ~r.review).sum()),
            }
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(OUT / "alert_replay_rows.csv", index=False)
    summary["rule_engine_runs"] = decide.cache_info().currsize
    (OUT / "alert_replay.json").write_text(json.dumps(summary, indent=2))
    cols = ["priority_correct", "high_recall", "high_precision", "false_high", "missed_high",
            "missed_high_sent_to_review", "review_rate", "errors_caught_by_review", "silent_priority_errors"]
    for key in ("results_before_fix", "results"):
        print(f"\n{key}:")
        print(pd.DataFrame(summary[key]).T[cols].to_string())
    print(f"\n{summary['rule_engine_runs']} distinct rule-engine runs; wrote {OUT / 'alert_replay.json'}")



def plot() -> None:
    """False HIGH alerts vs review workload, one line per classifier, points = review thresholds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.perception.camera_trap.plots import SERIES, T, style

    style()
    s = json.loads((OUT / "alert_replay.json").read_text())["results"]
    names = {"zero_shot": ("Zero-shot", SERIES["zero_shot_taxonomic"]),
             "logistic_regression": ("Logistic regression", SERIES["logistic_regression"]),
             "knn": ("kNN", SERIES["knn"])}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for clf, (label, color) in names.items():
        pts = sorted((v["review_rate"] * 100, v["false_high"], v["threshold"]) for v in s.values() if v["classifier"] == clf)
        xs, ys, ts = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6, markeredgecolor=T["surface"],
                markeredgewidth=1.2, label=label)
        for x, y, t in pts:
            ax.annotate(f"{t:g}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8, color=T["text2"])
    ax.set_xlabel("Sightings sent to human review (%)")
    ax.set_ylabel("False HIGH alerts (of 296 non-tiger photos)")
    ax.set_ylim(bottom=-0.6)
    ax.set_title("Review threshold vs false HIGH alerts", loc="left")
    ax.legend(loc="upper right", fontsize=9)
    fig.text(0.01, 0.01, "Point labels = review threshold. 334 held-out photos replayed through the rules, each with a person detected 1 h later.",
             fontsize=7.5, color=T["text2"])
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "alert_tradeoff.png", dpi=160)
    plt.close(fig)
    print(f"wrote {OUT / 'alert_tradeoff.png'}")


if __name__ == "__main__":
    main()
    plot()
