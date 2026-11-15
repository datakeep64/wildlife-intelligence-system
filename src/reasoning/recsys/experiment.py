"""Recommender comparison on held-out iNaturalist observers (SG + Peninsular Malaysia).

Protocol (no test-set peeking):
  validation: train < 2025-07-01, evaluate on 2025-07-01 .. 2025-12-31
              -> tune ALS, and choose the switching hybrid's model per segment
  test:       train < 2026-01-01, evaluate on records from 2026 (each model refitted once)
Target per user = species recorded after the cutoff that are NEW to them.
Segments: cold (no history), light (1-4 species), heavy (5+ species).

Outputs: results/recsys/metrics.json, per_user_test.csv, comparison.png
Usage:   .venv/bin/python -m src.reasoning.recsys.experiment
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # implicit's ALS is slower with multithreaded BLAS

import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.paths import DATA, RESULTS  # noqa: E402
from src.reasoning.recsys.base import PopularityRecommender  # noqa: E402
from src.reasoning.recsys.collaborative import ALSRecommender  # noqa: E402
from src.reasoning.recsys.content_based import ContentBasedRecommender, bioclip_features, taxonomy_features  # noqa: E402
from src.reasoning.recsys.data import item_popularity, load_records, make_split, novelty  # noqa: E402
from src.reasoning.recsys.hybrid import KnowledgeAdapter, RegionalPopularity, SwitchingHybrid, kg_to_catalogue  # noqa: E402

OUT = RESULTS / "recsys"
CACHE = DATA / "cache"
K = 10
VAL_CUTOFF, TEST_CUTOFF = "2025-07-01", "2026-01-01"
SEGMENTS = ["cold", "light", "heavy"]
ALS_GRID = {"factors": [8, 16, 32, 64], "regularization": [0.1, 1.0, 10.0, 100.0], "alpha": [0.3, 1.0, 10.0]}


def evaluate(model, split, pop, k=K) -> pd.DataFrame:
    rows = []
    for u, rel in split.truth.items():
        recs = [r.item for r in model.recommend(user=u, k=k, context=split.context[u], exclude=split.seen.get(u, set()))]
        hits = len(set(recs) & rel)
        rows.append({"user": u, "segment": split.segment[u], "served": len(recs) > 0, "precision": hits / k,
                     "recall": hits / len(rel), "hit": hits > 0, "novelty": novelty(recs, pop), "recs": "|".join(recs)})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, n_items: int, n_boot: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for seg, d in [("all", df)] + [(s, df[df.segment == s]) for s in SEGMENTS]:
        if d.empty:
            continue
        p = d.precision.to_numpy()
        boots = [p[rng.integers(0, len(p), len(p))].mean() for _ in range(n_boot)]
        recs = set(itertools.chain.from_iterable(r.split("|") for r in d.recs if r))
        out[seg] = {"users": int(len(d)), "precision@10": round(float(p.mean()), 4),
                    "precision_ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
                    "recall@10": round(float(d.recall.mean()), 4), "hit_rate@10": round(float(d.hit.mean()), 4),
                    "served": round(float(d.served.mean()), 4), "novelty_bits": round(float(d.novelty.mean()), 2),
                    "coverage": round(len(recs) / n_items, 4)}
    return out


def build(records, split, bio_feats, als_params) -> dict:
    train_records = records[records.date < split.cutoff]
    regional = RegionalPopularity(records).fit(split.train)
    models = {
        "popularity_global": PopularityRecommender().fit(split.train),
        "popularity_regional": regional,
        "knowledge_pure": KnowledgeAdapter("pure").fit(split.train),
        "knowledge_popularity": KnowledgeAdapter("popularity", regional).fit(split.train),
        "collaborative_als": ALSRecommender(**als_params).fit(split.train),
        "content_taxonomy": ContentBasedRecommender(taxonomy_features(records, train_records), "content_taxonomy").fit(split.train),
        "content_bioclip": ContentBasedRecommender(bio_feats, "content_bioclip").fit(split.train),
    }
    for name, m in models.items():
        m.name = name
    return models


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    records = load_records()
    items = sorted(records["item"].unique())

    bio_path = CACHE / "bioclip_species_text.pkl"
    if bio_path.exists() and set(pd.read_pickle(bio_path).index) >= set(items):
        bio = pd.read_pickle(bio_path).loc[items]
    else:
        bio = bioclip_features(items)
        bio.columns = [str(c) for c in bio.columns]
        bio.to_pickle(bio_path)

    # ------------------------------------------------ validation: tune ALS, choose hybrid policy
    val = make_split(records, VAL_CUTOFF, end=TEST_CUTOFF)
    val_pop = item_popularity(val.train)
    warm = {u for u, s in val.segment.items() if s != "cold"}
    als_scores = {}
    for f, r, a in itertools.product(*ALS_GRID.values()):
        m = ALSRecommender(factors=f, regularization=r, alpha=a).fit(val.train)
        res = evaluate(m, val, val_pop)
        als_scores[f"factors={f},reg={r},alpha={a}"] = round(float(res[res.user.isin(warm)].precision.mean()), 4)
    best = max(als_scores, key=als_scores.get)
    als_params = {k: float(v) if "." in v else int(v) for k, v in (kv.split("=") for kv in best.split(","))}
    als_params = {"factors": als_params["factors"], "regularization": als_params["reg"], "alpha": als_params["alpha"]}
    print(f"ALS best on validation: {best} -> {als_scores[best]}")

    val_models = build(records, val, bio, als_params)
    val_res = {n: summarize(evaluate(m, val, val_pop), len(items)) for n, m in val_models.items()}
    policy = {s: max(val_res, key=lambda n: val_res[n].get(s, {}).get("precision@10", -1)) for s in SEGMENTS}
    print("hybrid policy (chosen on validation):", policy)

    # ------------------------------------------------ test: refit once, evaluate once
    test = make_split(records, TEST_CUTOFF)
    pop = item_popularity(test.train)
    models = build(records, test, bio, als_params)
    models["switching_hybrid"] = SwitchingHybrid(models, policy, test.segment)
    per_user, results = [], {}
    for name, m in models.items():
        df = evaluate(m, test, pop).assign(model=name)
        per_user.append(df)
        results[name] = summarize(df, len(items))
        a = results[name]
        print(f"  {name:22s} " + "  ".join(f"{s}: P@10={a[s]['precision@10']:.3f}" for s in ["all"] + SEGMENTS if s in a))
    # observer names are public on GBIF, but a public repo doesn't need them: one-way hash
    pu = pd.concat(per_user).drop(columns=["recs"])
    pu["user"] = pu.user.map(lambda u: hashlib.sha256(u.encode()).hexdigest()[:12])
    pu.to_csv(OUT / "per_user_test.csv", index=False)

    kg_items = set(kg_to_catalogue().values())
    truth_items = list(itertools.chain.from_iterable(test.truth.values()))
    seg_counts = pd.Series(test.segment).value_counts().to_dict()
    metrics = {
        "protocol": {"validation": f"train < {VAL_CUTOFF}, evaluate {VAL_CUTOFF}..{TEST_CUTOFF}",
                     "test": f"train < {TEST_CUTOFF}, evaluate >= {TEST_CUTOFF}", "k": K,
                     "target": "species recorded after the cutoff that are new to the user"},
        "data": {"records": int(len(records)), "items": len(items), "train_users": int(test.train.user.nunique()),
                 "test_users": len(test.truth), "test_users_by_segment": seg_counts,
                 "kg_species_share_of_test_targets": round(float(np.mean([i in kg_items for i in truth_items])), 4)},
        "als_validation_grid": als_scores, "als_params": als_params,
        "hybrid_policy": policy, "validation": val_res, "test": results,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote {OUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
