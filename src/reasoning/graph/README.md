# Property graph and link prediction

**Status: in progress.** The property graph and its analytics are done. Link prediction code and tests are done; tuning is paused halfway (see [STATUS.md](../../../STATUS.md)). No test results yet.

## Question
A ranger's map only shows where species have been *recorded*, and survey effort is very uneven. Can the graph's structure predict where a species will be recorded next? That would help decide where to put cameras.

## Property graph (`property_graph.py`)
- GBIF records (iNaturalist observers + the curated species' own download) are binned into 0.1° squares (~11 km).
- **Precision guard:** records blurred by more than half a square are dropped. That is 28% of records, and most threatened-species records.
- **Nodes:** species, squares, families, orders, classes, regions, IUCN statuses, and habitats (from the curated ontology KG).
- **Edges:** `recordedIn` (with records, observers and first year), `adjacentTo`, `inRegion`, the taxonomy chain, `hasStatus` and `hasHabitat`.
- **Why a property graph and not OWL:** the questions here are graph analytics (effort, co-occurrence, neighbourhoods), not logical entailment.

**Finding so far:** survey effort is extremely concentrated. 89% of records come from 10% of squares, and a square's species count correlates 0.87 with its (log) record count. "Species-rich" mostly means "well surveyed".

## Link prediction (`link_prediction.py`, `embeddings.py`, `experiment.py`)
- **Task:** predict species → square links **first recorded in 2025–26**, training on everything known by the end of 2024. Tuning uses the same setup one year earlier (train ≤ 2023, predict 2024).
- **Scoring:** filtered ranking against every square where the species hasn't been recorded, with ties counted half. Metrics are Hits@10 (primary: "which 10 squares should we survey?"), MRR and Hits@50. 95% intervals come from resampling species.
- **Models:** TransE (relation = translation) vs RotatE (relation = rotation, so it can represent symmetric relations such as `adjacentTo`). Each tries both standard losses, and validation chooses.
- **Baselines:**
  - **popularity:** the squares with the most species
  - **co-occurrence neighbours:** item-kNN over shared squares
  - **spatial neighbour:** the species is already in an adjacent square
- **Ablation (planned):** drop the spatial, taxonomy or species-fact relations to see which knowledge the embeddings use.
- **Limits:**
  - Embeddings can't score a species or square that isn't in the training graph. Those links are counted and reported.
  - Threatened species are left out of every per-square output.

**Preliminary validation numbers** (single untuned runs; not results), Hits@10:

| Method | Hits@10 |
|---|---|
| Popularity | 0.17 |
| Spatial neighbour | 0.20 |
| Co-occurrence | 0.24 |
| Best TransE run so far | 0.26 |

## Resume
```
# first extend EPOCHS in experiment.py (9 of 24 TransE runs peaked at the last epoch, 150)
python -m src.reasoning.graph.experiment     # ~45 min CPU; finished tuning runs are cached in data/cache/
```
