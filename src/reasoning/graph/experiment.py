"""Property graph + link prediction experiment.

  1. Property graph analytics (survey effort, co-occurrence).
  2. Validation (train <= 2023, predict links first recorded in 2024): tune every method.
  3. Test (train <= 2024, predict links first recorded in 2025-2026): chosen settings, 5 seeds.
  4. Ablation: which kinds of knowledge do the embeddings use?

Run: python -m src.reasoning.graph.experiment
Writes results/graph/. Threatened species (VU/EN/CR) are left out of every per-square output.
"""
import itertools
import json
import logging
import time

import numpy as np
import pandas as pd

from src.common.paths import DATA, RESULTS
from .embeddings import KGE
from .link_prediction import (cooccurrence_scores, make_split, metrics, popularity_scores, rank_test,
                              spatial_scores)
from .property_graph import (RELATION_GROUPS, THREATENED, build_graph, cooccurrence, effort_summary,
                             load_occurrences, pairs, species_info, triples)

OUT = RESULTS / "graph"
CACHE = DATA / "cache" / "graph_val_grid.json"   # resumable: finished tuning runs are skipped
logging.getLogger("pykeen").setLevel(logging.ERROR)

KNN_K = [5, 10, 25, 50, 100, 10_000]
GRID = {"dim": [32, 64, 128], "lr": [0.002, 0.01], "loss": ["margin", "self_adversarial"]}
EPOCHS = [10, 25, 50, 100, 150]
VAL_SEEDS, TEST_SEEDS, ABL_SEEDS = (0, 1), (0, 1, 2, 3, 4), (0, 1, 2)
PRIMARY = "hits@10"          # the use case: "which 10 squares should we survey for this species?"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tune_embeddings(split, info, cache: dict) -> dict:
    tri = triples(split.train, info)
    for model, (dim, lr, loss), seed in itertools.product(("TransE", "RotatE"), itertools.product(*GRID.values()),
                                                          VAL_SEEDS):
        key = f"{model}|{dim}|{lr}|{loss}|{seed}"
        if key in cache:
            continue
        k = KGE(model, tri, dim=dim, lr=lr, loss=loss, seed=seed)
        cache[key] = {ep: metrics(rank_test(split, k.train_to(ep).scores(split))) for ep in EPOCHS}
        log("val", key, {ep: round(m[PRIMARY], 3) for ep, m in cache[key].items()})
        (CACHE).write_text(json.dumps(cache))
    best = {}
    for model in ("TransE", "RotatE"):
        rows = []
        for dim, lr, loss in itertools.product(*GRID.values()):
            for ep in EPOCHS:
                ms = [cache[f"{model}|{dim}|{lr}|{loss}|{s}"][ep] for s in VAL_SEEDS]
                rows.append({"dim": dim, "lr": lr, "loss": loss, "epochs": ep,
                             PRIMARY: float(np.mean([m[PRIMARY] for m in ms])),
                             "mrr": float(np.mean([m["mrr"] for m in ms]))})
        df = pd.DataFrame(rows).sort_values([PRIMARY, "mrr"], ascending=False)
        best[model] = {"config": df.iloc[0][["dim", "lr", "loss", "epochs"]].to_dict(),
                       "val": df.iloc[0][[PRIMARY, "mrr"]].to_dict(),
                       "by_loss": df.groupby("loss")[PRIMARY].max().to_dict()}
        best[model]["config"]["dim"] = int(best[model]["config"]["dim"])
        best[model]["config"]["epochs"] = int(best[model]["config"]["epochs"])
    return best


def per_link(split, ranks_by_seed: list[pd.DataFrame]) -> pd.DataFrame:
    """Average each link's reciprocal rank and hit over seeds (for paired, species-clustered bootstrap)."""
    base = ranks_by_seed[0][["species", "cell"]].copy()
    r = np.stack([x["rank"].to_numpy() for x in ranks_by_seed])
    base["rr"] = (1 / r).mean(0)
    base["hit10"] = (r <= 10).mean(0)
    base["rank"] = np.median(r, 0)
    return base


def boot(values: pd.Series, species: pd.Series, n=2000, seed=0) -> list[float]:
    g = pd.DataFrame({"s": species.values, "v": values.values}).groupby("s").v.agg(["sum", "size"])
    S, N = g["sum"].to_numpy(), g["size"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n, len(g)))
    st = S[idx].sum(1) / N[idx].sum(1)
    return [round(float(np.percentile(st, 2.5)), 4), round(float(np.percentile(st, 97.5)), 4)]


def segments(split, info) -> pd.DataFrame:
    X = split.incidence()
    sp_per_cell = dict(zip(split.cells, X.sum(0)))
    range_size = dict(zip(split.species, X.sum(1)))
    t = split.test[["species", "cell"]].copy()
    n = t.cell.map(sp_per_cell)
    t["square"] = np.where(n < 5, "sparse (<5 species)", np.where(n < 20, "medium (5-19)", "dense (20+)"))
    rs = t.species.map(range_size)
    t["species_range"] = np.where(rs < 3, "narrow (1-2 squares)", np.where(rs < 10, "mid (3-9)", "wide (10+)"))
    t["threatened"] = t.species.map(info.iucn).isin(THREATENED)
    return t


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = load_occurrences()
    info = species_info(d)
    summary = {"records_kept": int(len(d)), "records_dropped_imprecise": int(d.attrs["dropped_imprecise"]),
               "species": int(d.species.nunique())}

    # ---------------------------------------------------------- 1. property graph analytics
    links_all = pairs(d)
    g = build_graph(links_all, info)
    co = cooccurrence(links_all)
    co["threatened_pair"] = co.a.map(info.iucn).isin(THREATENED) | co.b.map(info.iucn).isin(THREATENED)
    summary["property_graph"] = {
        "nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
        "species_square_links": int(len(links_all)),
        "effort": effort_summary(d, links_all),
        "cooccurring_pairs_min3": int(len(co)),
    }
    co[~co.threatened_pair].head(30).drop(columns="threatened_pair").to_csv(OUT / "top_cooccurrence.csv", index=False)
    log("property graph", summary["property_graph"])

    # ---------------------------------------------------------- 2. validation: tune
    val = make_split(d, 2023, (2024,))
    summary["validation_split"] = {"label": val.label, "train_links": len(val.train), "test_links": len(val.test),
                                   **val.excluded}
    knn_val = {k: metrics(rank_test(val, cooccurrence_scores(val, k)))[PRIMARY] for k in KNN_K}
    best_k = max(knn_val, key=knn_val.get)
    cache_f = CACHE
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    cache = {k: {int(e): m for e, m in v.items()} for k, v in cache.items()}
    best = tune_embeddings(val, info, cache)
    summary["tuning"] = {"cooccurrence_k": {"chosen": best_k, "val": knn_val}, "embeddings": best}
    log("tuned", best_k, best)

    # ---------------------------------------------------------- 3. test
    test = make_split(d, 2024, (2025, 2026))
    summary["test_split"] = {"label": test.label, "train_links": len(test.train), "test_links": len(test.test),
                             **test.excluded}
    ranks = {"Popularity": [rank_test(test, popularity_scores(test))],
             "Spatial neighbour": [rank_test(test, spatial_scores(test))],
             "Co-occurrence": [rank_test(test, cooccurrence_scores(test, best_k))]}
    tri = triples(test.train, info)
    for model in ("TransE", "RotatE"):
        c = best[model]["config"]
        ranks[model] = []
        for s in TEST_SEEDS:
            k = KGE(model, tri, dim=c["dim"], lr=c["lr"], loss=c["loss"], seed=s).train_to(c["epochs"])
            ranks[model].append(rank_test(test, k.scores(test)))
            log("test", model, s, round(metrics(ranks[model][-1])[PRIMARY], 3))
    links = {m: per_link(test, r) for m, r in ranks.items()}
    seg = segments(test, info)
    res = {}
    for m, r in ranks.items():
        per_seed = [metrics(x) for x in r]
        L = links[m]
        res[m] = {k: round(float(np.mean([p[k] for p in per_seed])), 4) for k in ("mrr", "hits@10", "hits@50", "median_rank")}
        res[m]["seed_sd_hits@10"] = round(float(np.std([p["hits@10"] for p in per_seed])), 4)
        res[m]["ci_hits@10"] = boot(L.hit10, L.species)
        res[m]["ci_mrr"] = boot(L.rr, L.species)
        res[m]["by_segment"] = {col: L.assign(seg=seg[col].values).groupby("seg").hit10.mean().round(4).to_dict()
                                for col in ("square", "species_range", "threatened")}
    summary["test"] = res
    summary["segment_sizes"] = {col: seg[col].astype(str).value_counts().to_dict()
                                for col in ("square", "species_range", "threatened")}
    best_base = max(("Popularity", "Spatial neighbour", "Co-occurrence"), key=lambda m: res[m]["hits@10"])
    summary["paired_vs_best_baseline"] = {
        "baseline": best_base,
        **{m: {"diff_hits@10": round(float((links[m].hit10 - links[best_base].hit10).mean()), 4),
               "ci": boot(links[m].hit10 - links[best_base].hit10, links[m].species)}
           for m in ("TransE", "RotatE", *[b for b in ranks if b != best_base and b not in ("TransE", "RotatE")])}}
    log("test results", json.dumps(res, indent=1))

    # public per-link file: no threatened species
    out = seg[["species", "cell", "square", "species_range"]].copy()
    for m in links:
        out[f"rank_{m}"] = links[m]["rank"].values
    out[~seg.threatened].to_csv(OUT / "test_ranks.csv", index=False)

    # ---------------------------------------------------------- 4. ablation (test split)
    abl_sets = {"all knowledge": tuple(RELATION_GROUPS),
                "no space (adjacency, region)": ("occurrence", "taxonomy", "species_facts"),
                "no taxonomy": ("occurrence", "space", "species_facts"),
                "no species facts (status, habitat)": ("occurrence", "space", "taxonomy"),
                "occurrence links only": ("occurrence",)}
    abl = {}
    for model in ("TransE", "RotatE"):
        c = best[model]["config"]
        abl[model] = {}
        for name, groups in abl_sets.items():
            if name == "all knowledge":
                rs = ranks[model][:len(ABL_SEEDS)]
            else:
                tri_a = triples(test.train, info, groups)
                rs = [rank_test(test, KGE(model, tri_a, dim=c["dim"], lr=c["lr"], loss=c["loss"], seed=s)
                                .train_to(c["epochs"]).scores(test)) for s in ABL_SEEDS]
            hs = [metrics(x)["hits@10"] for x in rs]
            abl[model][name] = {"hits@10": round(float(np.mean(hs)), 4), "sd": round(float(np.std(hs)), 4)}
            log("ablation", model, name, abl[model][name])
    summary["ablation_hits@10"] = abl

    # ---------------------------------------------------------- 5. worked example for the README / plots
    best_model = max(("TransE", "RotatE"), key=lambda m: res[m]["hits@10"])
    c = best[best_model]["config"]
    k = KGE(best_model, tri, dim=c["dim"], lr=c["lr"], loss=c["loss"], seed=0).train_to(c["epochs"])
    S = k.scores(test)
    X = test.incidence().astype(bool)
    rows = []
    for s in test.test.species.unique():
        if info.iucn.get(s) in THREATENED:
            continue
        i = test.s_idx[s]
        sc = np.where(X[i], -np.inf, S[i])
        top = [test.cells[j] for j in np.argsort(-sc)[:10]]
        truth = set(test.test[test.test.species == s].cell)
        rows.append({"species": s, "train_squares": int(X[i].sum()), "new_squares_2025_26": len(truth),
                     "top10_hits": len(truth & set(top)), "top10": ";".join(top), "truth": ";".join(sorted(truth))})
    pd.DataFrame(rows).sort_values(["top10_hits", "new_squares_2025_26"], ascending=False).to_csv(
        OUT / "survey_suggestions_examples.csv", index=False)

    summary["runtime_min"] = round((time.time() - t0) / 60, 1)
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    log("done", summary["runtime_min"], "min")


if __name__ == "__main__":
    main()
