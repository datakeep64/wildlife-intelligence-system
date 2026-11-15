"""Camera-trap experiment: zero-shot BioCLIP vs linear probe vs kNN on frozen embeddings.

Question: is BioCLIP's representation already good enough for SE Asian camera
traps, or does a small labelled set change the answer?

Stages (each cached in data/lila_wcs/cache/, so reruns are cheap):
  1. MegaDetector on every image          -> triage metrics (animal recall, empty specificity)
  2. BioCLIP embeddings of crop AND full image, + text prompts (taxonomic and plain)
  3. Classifiers on the location-grouped split:
       zero-shot (taxonomic prompts, plain prompts, full-image ablation)
       logistic regression and kNN (hyper-parameters picked by GroupKFold on train locations)
  4. Uncertainty: cluster bootstrap over camera LOCATIONS (images from one camera are
     not independent, so an image-level bootstrap would give falsely narrow intervals)
  5. Few-shot curve, leopard Kenya-vs-SE-Asia breakdown, pangolin open-set test
  6. Embedding geometry: silhouette score + UMAP coordinates for the figure

Outputs: results/camera_trap/metrics.json, predictions_test.csv, umap_coords.csv
Figures: python -m src.perception.camera_trap.plots
Usage:   .venv/bin/python -m src.perception.camera_trap.experiment
"""
import json
import re

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, silhouette_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from src.common.paths import LILA_DIR, RESULTS, SPECIES_CSV
from src.perception.camera_trap.classify import KNNProbe, LinearProbe, evaluate, zero_shot
from src.perception.camera_trap.dataset import load_config
from src.perception.camera_trap.detect import Detector, best_box, crop

CACHE = LILA_DIR / "cache"
OUT = RESULTS / "camera_trap"
N_BOOT = 1000
FEW_SHOTS = [1, 2, 5, 10, 20, 50]
FEW_SEEDS = 10


# ------------------------------------------------------------------ stage 1
def run_detection(df: pd.DataFrame, cfg: dict) -> dict:
    path = CACHE / "detections.json"
    dets = json.loads(path.read_text()) if path.exists() else {}
    todo = [r for r in df.itertuples() if str(r.image_id) not in dets]
    if todo:
        det = Detector(cfg["md_model"], cfg["md_threshold"])
        for i, r in enumerate(tqdm(todo, desc="MegaDetector")):
            dets[str(r.image_id)] = det.detect(r.local_path)
            if i % 50 == 0:
                path.write_text(json.dumps(dets))
        path.write_text(json.dumps(dets))
    return dets


def triage_metrics(df: pd.DataFrame, dets: dict) -> dict:
    d = df.assign(
        has_animal=[any(x["category"] == "animal" for x in dets[str(i)]["detections"]) for i in df.image_id],
        has_person=[any(x["category"] == "person" for x in dets[str(i)]["detections"]) for i in df.image_id],
        has_vehicle=[any(x["category"] == "vehicle" for x in dets[str(i)]["detections"]) for i in df.image_id],
        n_dets=[len(dets[str(i)]["detections"]) for i in df.image_id],
    )
    animals = d[d.role != "triage_empty"]
    empty = d[d.role == "triage_empty"]
    return {
        "threshold": 0.2,
        "animal_images": int(len(animals)),
        "animal_recall": round(float(animals.has_animal.mean()), 4),
        "animal_recall_per_class": animals.groupby("label").has_animal.mean().round(4).to_dict(),
        "false_person_on_animal_images": int(animals.has_person.sum()),
        "empty_images": int(len(empty)),
        "empty_specificity": round(float((empty.n_dets == 0).mean()), 4),
        "false_animal_on_empty": int(empty.has_animal.sum()),
        "false_person_on_empty": int(empty.has_person.sum()),
        "false_vehicle_on_empty": int(empty.has_vehicle.sum()),
    }


# ------------------------------------------------------------------ stage 2
def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def run_embeddings(df: pd.DataFrame, dets: dict, cfg: dict, prompt_sets: dict[str, list[str]]):
    path = CACHE / f"embeddings_{slug(cfg['model'])}.npz"
    ids = df.image_id.astype(str).tolist()
    if path.exists():
        z = np.load(path, allow_pickle=True)
        if z["ids"].tolist() == ids and set(prompt_sets) <= set(z["text_names"].tolist()):
            text = {n: z[f"text_{n}"] for n in prompt_sets}
            return z["crop"], z["full"], z["md_miss"], text
    from src.perception.camera_trap.embed import BioCLIP

    clip = BioCLIP(cfg["model"])
    print(f"BioCLIP on {clip.device}")
    crops, fulls, miss = [], [], []
    bc, bf = [], []
    for r in tqdm(list(df.itertuples()), desc="BioCLIP"):
        im = Image.open(r.local_path).convert("RGB")
        box = best_box(dets[str(r.image_id)]["detections"], "animal")
        miss.append(box is None)
        bf.append(im)
        bc.append(crop(im, box["bbox"]) if box else im)  # MD miss -> classify full frame
        if len(bc) == 32:
            crops.append(clip.image_embeddings(bc)); fulls.append(clip.image_embeddings(bf)); bc, bf = [], []
    if bc:
        crops.append(clip.image_embeddings(bc)); fulls.append(clip.image_embeddings(bf))
    text = {n: clip.text_embeddings(p) for n, p in prompt_sets.items()}
    crop_e, full_e, md_miss = np.concatenate(crops), np.concatenate(fulls), np.array(miss)
    np.savez(path, ids=np.array(ids, dtype=object), crop=crop_e, full=full_e, md_miss=md_miss,
             text_names=np.array(list(text), dtype=object), **{f"text_{n}": v for n, v in text.items()})
    return crop_e, full_e, md_miss, text


# ------------------------------------------------------------------ stage 3
def select(make, grid, X, y, groups) -> tuple:
    """Pick a hyper-parameter by macro-F1 under GroupKFold over camera locations (train only)."""
    gkf = GroupKFold(n_splits=5)
    scores = {}
    for g in grid:
        s = [f1_score(y[va], make(g).fit(X[tr], y[tr]).predict(X[va]), average="macro")
             for tr, va in gkf.split(X, y, groups)]
        scores[g] = round(float(np.mean(s)), 4)
    return max(scores, key=scores.get), scores


def cluster_bootstrap(y, preds: dict, groups, labels, n=N_BOOT, seed=0) -> dict:
    """Resample camera locations with replacement; report 95% CIs and paired differences."""
    rng = np.random.default_rng(seed)
    locs = np.unique(groups)
    by_loc = {l: np.where(groups == l)[0] for l in locs}
    stats = {k: {"acc": [], "f1": []} for k in preds}
    for _ in range(n):
        idx = np.concatenate([by_loc[l] for l in rng.choice(locs, size=len(locs), replace=True)])
        for k, p in preds.items():
            stats[k]["acc"].append((p[idx] == y[idx]).mean())
            stats[k]["f1"].append(f1_score(y[idx], p[idx], labels=labels, average="macro", zero_division=0))
    out = {}
    for k, s in stats.items():
        out[k] = {m: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for m, v in s.items()}
    base = "zero_shot_taxonomic"
    out["paired_diff_vs_zero_shot"] = {}
    for k in preds:
        if k == base:
            continue
        d = np.array(stats[k]["f1"]) - np.array(stats[base]["f1"])
        out["paired_diff_vs_zero_shot"][k] = {
            "macro_f1_diff_ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "share_of_resamples_better": round(float((d > 0).mean()), 4),
        }
    out["n_test_locations"] = int(len(locs))
    return out


def few_shot_curve(Xtr, ytr, Xte, yte, C, k, labels) -> list[dict]:
    rows = []
    for n in FEW_SHOTS:
        for seed in range(FEW_SEEDS):
            rng = np.random.default_rng(seed)
            idx = np.concatenate([rng.choice(np.where(ytr == c)[0], size=min(n, (ytr == c).sum()), replace=False) for c in labels])
            for name, m in [("logistic_regression", LinearProbe(C=C)), ("knn", KNNProbe(k=min(k, n)))]:
                p = m.fit(Xtr[idx], ytr[idx]).predict(Xte)
                rows.append({"shots": n, "seed": seed, "model": name,
                             "macro_f1": f1_score(yte, p, labels=labels, average="macro", zero_division=0),
                             "accuracy": float((p == yte).mean())})
    return rows


# ------------------------------------------------------------------ main
def main() -> None:
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(LILA_DIR / "manifest.csv", dtype={"location": str, "image_id": str})
    labels = [c["key"] for c in cfg["classes"] if c["key"] in set(df.label)]  # 'human' has no images
    labels10 = labels + [c["key"] for c in cfg["zero_shot_only"]]
    common = pd.read_csv(SPECIES_CSV, dtype=str).set_index("key")["common_en"]
    prompt_sets = {
        "taxonomic9": [cfg["prompts"][k] for k in labels],
        "plain9": [f"a camera trap photo of a {common[k]}" for k in labels],
        "taxonomic10": [cfg["prompts"][k] for k in labels10],
    }
    print(f"{len(df)} images; classes: {labels}")

    dets = run_detection(df, cfg)
    triage = triage_metrics(df, dets)
    print("triage:", {k: v for k, v in triage.items() if not isinstance(v, dict)})

    crop_e, full_e, md_miss, text = run_embeddings(df, dets, cfg, prompt_sets)
    df["md_miss"] = md_miss

    exp = (df.role == "experiment").to_numpy()
    tr = exp & (df.split == "train").to_numpy()
    te = exp & (df.split == "test").to_numpy()
    y = df.label.to_numpy()
    groups = df.location.fillna("unknown").to_numpy()

    # ---- classifiers
    C, c_scores = select(lambda c: LinearProbe(C=c), [0.1, 1.0, 10.0, 100.0], crop_e[tr], y[tr], groups[tr])
    k, k_scores = select(lambda k: KNNProbe(k=k), [1, 3, 5, 10, 20], crop_e[tr], y[tr], groups[tr])
    print(f"selected C={C} {c_scores} | k={k} {k_scores}")
    preds = {
        "zero_shot_taxonomic": zero_shot(crop_e[te], text["taxonomic9"], labels)[0],
        "zero_shot_plain": zero_shot(crop_e[te], text["plain9"], labels)[0],
        "zero_shot_full_image": zero_shot(full_e[te], text["taxonomic9"], labels)[0],
        "logistic_regression": LinearProbe(C=C).fit(crop_e[tr], y[tr]).predict(crop_e[te]),
        "logistic_regression_full_image": LinearProbe(C=C).fit(full_e[tr], y[tr]).predict(full_e[te]),
        "knn": KNNProbe(k=k).fit(crop_e[tr], y[tr]).predict(crop_e[te]),
    }
    results = {name: evaluate(y[te], p, labels) for name, p in preds.items()}
    for name, r in results.items():
        print(f"  {name:32s} acc={r['accuracy']:.3f}  macroF1={r['macro_f1']:.3f}")

    boot = cluster_bootstrap(y[te], preds, groups[te], labels)

    # ---- MD misses: were they classified worse?
    miss_te = md_miss[te]
    md_effect = {name: {"acc_md_hit": round(float((p[~miss_te] == y[te][~miss_te]).mean()), 4),
                        "acc_md_miss": round(float((p[miss_te] == y[te][miss_te]).mean()), 4) if miss_te.any() else None}
                 for name, p in preds.items() if name in ("zero_shot_taxonomic", "logistic_regression")}
    md_effect["n_md_miss_test"] = int(miss_te.sum())

    # ---- leopard domain gap (Kenya vs SE Asia) on test
    leo = te & (y == "leopard")
    leopard = {}
    for ctry in sorted(set(df.country_code[leo])):
        m = (df.country_code[te] == ctry).to_numpy() & (y[te] == "leopard")
        leopard[ctry] = {"n": int(m.sum()), **{name: round(float((preds[name][m] == "leopard").mean()), 4)
                                                 for name in ("zero_shot_taxonomic", "logistic_regression", "knn")}}
    leo_train = df[tr & (y == "leopard")].country_code.value_counts().to_dict()

    # ---- pangolin: open-set test (zero-shot can name it; closed-set probes cannot)
    pang = (df.label == "sunda_pangolin").to_numpy()
    zs10_p = zero_shot(crop_e[pang], text["taxonomic10"], labels10)[0]
    lr_p = LinearProbe(C=C).fit(crop_e[tr], y[tr]).predict(crop_e[pang])
    zs10_test = zero_shot(crop_e[te], text["taxonomic10"], labels10)[0]
    pangolin = {
        "n": int(pang.sum()),
        "zero_shot_10_labels_recall": round(float((zs10_p == "sunda_pangolin").mean()), 4),
        "zero_shot_10_labels_predictions": pd.Series(zs10_p).value_counts().to_dict(),
        "logistic_regression_forced_into": pd.Series(lr_p).value_counts().to_dict(),
        "false_pangolin_on_9class_test": int((zs10_test == "sunda_pangolin").sum()),
        "zero_shot_9class_test_acc_with_pangolin_label": round(float((zs10_test == y[te]).mean()), 4),
    }

    # ---- few-shot curve
    fs = pd.DataFrame(few_shot_curve(crop_e[tr], y[tr], crop_e[te], y[te], C, k, labels))
    fs.to_csv(OUT / "few_shot_curve.csv", index=False)
    fs_summary = fs.groupby(["model", "shots"]).macro_f1.agg(["mean", "std"]).round(4).reset_index().to_dict("records")

    # ---- embedding geometry
    from sklearn.decomposition import PCA
    import umap

    Xe = crop_e[exp]
    sil_crop = silhouette_score(Xe, y[exp], metric="cosine")
    sil_full = silhouette_score(full_e[exp], y[exp], metric="cosine")
    pca = PCA(n_components=10, random_state=42).fit(Xe)
    coords = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42).fit_transform(Xe)
    pd.DataFrame({"image_id": df.image_id[exp], "label": y[exp], "split": df.split[exp],
                  "country": df.country_code[exp], "x": coords[:, 0], "y": coords[:, 1]}).to_csv(OUT / "umap_coords.csv", index=False)

    # ---- save
    test_df = df[te][["image_id", "label", "country_code", "location", "md_miss"]].copy()
    for name, p in preds.items():
        test_df[name] = p
    test_df.to_csv(OUT / "predictions_test.csv", index=False)

    counts = df[exp].groupby(["label", "split"]).size().unstack(fill_value=0).to_dict("index")
    metrics = {
        "model": cfg["model"], "md_model": cfg["md_model"],
        "data": {"n_train": int(tr.sum()), "n_test": int(te.sum()), "per_class": counts,
                 "n_train_locations": int(len(set(groups[tr]))), "n_test_locations": int(len(set(groups[te])))},
        "triage": triage,
        "hyperparameters": {"logreg_C": C, "logreg_cv_macro_f1": c_scores, "knn_k": k, "knn_cv_macro_f1": k_scores},
        "results": results,
        "bootstrap_ci95_by_location": boot,
        "md_miss_effect": md_effect,
        "leopard_by_country_test": leopard, "leopard_train_countries": leo_train,
        "pangolin_open_set": pangolin,
        "few_shot_macro_f1": fs_summary,
        "embedding": {"silhouette_cosine_crop": round(float(sil_crop), 4), "silhouette_cosine_full_image": round(float(sil_full), 4),
                      "pca_explained_variance_top10": [round(float(v), 4) for v in pca.explained_variance_ratio_]},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nwrote {OUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
