# STATUS

_Last updated: 2026-11-15 (session 1: Phase A done; Phase B camera-trap experiment, alert rules and recommender comparison done; property graph + link prediction in progress, paused mid-tuning)_

## Components

| Component | Status | What exists |
|---|---|---|
| Repo scaffold | Done | Folders, `requirements.txt` + lockfile (Python 3.12 venv), `.gitignore`, README with Mermaid architecture |
| Data: IUCN status | Done | `data/download_iucn.py` → statuses for 26 taxa via GBIF |
| Data: LILA camera traps | Done | `data/download_lila.py` → 1,400 WCS images, location-grouped split |
| Data: GBIF occurrences | Done | `data/download_gbif.py`: 7,062 seed-species records + 65,953 iNaturalist observer records (year-stratified sample) |
| Ontology KG + alert rules | Done | TBox + seed ABox + GBIF reconciliation (`observations.ttl`); 18 SPARQL rules in 5 strata (forward chaining, stratified negation, conflict resolution, explanations); 17 scenario tests; error replay of real camera-trap predictions ([README](src/reasoning/kg/README.md)) |
| Camera-trap pipeline + experiment | Done | MegaDetector → crop → BioCLIP → zero-shot / logistic regression / kNN on 1,400 images, with a location-grouped split, location bootstrap, few-shot curve and 4 figures ([README](src/perception/camera_trap/README.md)). Person detection on COCO still to do |
| NLLB translation | Working (results pending) | Module (raw + glossary modes), 20-alert eval set, evaluator; smoke run on 3 alerts × 3 languages done ([outputs](results/translation/outputs.csv)) |
| Recommenders + comparison | Done | Popularity (global, by country), knowledge-based (2 variants), ALS, content (taxonomy, BioCLIP), switching hybrid; time-based split with a validation stage; per-segment results and 2 figures ([README](src/reasoning/recsys/README.md)). Trail suggestions built but not evaluated |
| Property graph | Working | `src/reasoning/graph/property_graph.py`: GBIF records → 0.1° squares (~11 km) → NetworkX graph (species, squares, taxonomy, region, IUCN status, curated habitats); survey-effort and co-occurrence analytics done |
| Embedding KG (PyKEEN) | In progress (paused) | Link prediction code, 3 baselines, filtered ranking and 5 tests done. Tuning paused at 24/48 validation runs (all TransE), cached and resumable. No test results yet ([README](src/reasoning/graph/README.md)) |
| Bioacoustics (optional) | Not started | |
| Integration loop | Not started | Phase C |
| Portfolio demo page | Not started | Phase C |

## Results so far (real, measured)

- **Camera-trap dataset** ([data/README.md](data/README.md)):
  - 9 classes × 133–150 images, 1 frame per burst sequence, plus 17 pangolin images (zero-shot only) and 50 empty frames.
  - The test split is grouped by camera location and holds 25% of each class.
- **Camera-trap experiment** (334 test photos from 160 camera locations unseen in training; 95% CIs from a location bootstrap; [metrics.json](results/camera_trap/metrics.json)):
  - **Zero-shot BioCLIP:** macro-F1 **0.822** (0.764–0.869).
  - **Logistic-regression probe:** macro-F1 **0.913** (0.878–0.945). **kNN probe:** **0.914** (0.881–0.944).
  - The probe's gain over zero-shot is +0.05 to +0.14 (paired), and it was better in 100% of resamples.
  - **Five labelled photos per species** are enough for logistic regression to beat zero-shot (0.862 vs 0.822); two are not (0.777). [few_shot_curve.png](results/camera_trap/few_shot_curve.png)
  - **Zero-shot errors flow into "attractor" classes, not into look-alikes.** Of 60 errors, 16 went to wild boar, 15 to tapir and 12 to tiger. The pre-registered look-alike pairs were mostly confused ≤6% of the time (exception: clouded leopard→tiger 9% for zero-shot). [confusion matrices](results/camera_trap/confusion_zero_shot_vs_probe.png)
  - **Cropping helps:** zero-shot 0.822 vs 0.778 on the full frame; probe 0.913 vs 0.871. **Taxonomic prompts beat plain prompts** (0.822 vs 0.777).
  - **MegaDetector at threshold 0.2:** 98.1% animal recall (1,350 animal photos); 88% of empty frames correctly empty (6/50 got a false animal box); 0 false person or vehicle alarms.
  - **Pangolin (open set):** zero-shot named 9/17 correctly. The closed-set probe forced 8/17 into "clouded leopard", because it can't say "new species".
  - **Leopards:** all 20 Lao leopards cluster with the Kenyan ones in embedding space, so the domain gap is small at the representation level (n=20).
- **Alert rules: error replay** (334 held-out photos → sightings, each followed by a person detection; 38 should be HIGH; [alert_replay.json](results/kg/alert_replay.json)):
  - **Zero-shot, no review:** 12 false HIGH alerts (HIGH precision 0.76); 5 of them are tapirs called tigers.
  - **Logistic regression with review at 0.5:** 0 false HIGH (precision 1.00, recall 0.95), with 6% of sightings sent to review.
  - **Zero-shot is confidently wrong** (22% of its errors at ≥ 0.9 confidence vs 3% for logistic regression). Even with 17% of sightings reviewed, it still raises 3 false HIGH alerts. [alert_tradeoff.png](results/kg/alert_tradeoff.png)
  - **Review blind spot, found and fixed:** a tiger misread as *wild boar* at 0.34 confidence passed silently, because review only triggered when the *predicted* species was threatened. New rule C4 also reviews low-confidence sightings whose *runner-up* is threatened. For logistic regression at 0.5: silently missed tigers 1 → **0**, silent priority errors 12 → 7, errors caught by review 38% → 55%, at a cost of 6.3% → 8.7% of sightings reviewed. False HIGH alerts stay at 0.
- **Recommender comparison** (1,869 held-out 2026 observers: 1,387 cold, 259 light, 223 heavy; precision@10; [metrics.json](results/recsys/metrics.json)):
  - **Regional popularity won:** 0.134 (cold), 0.129 (light), 0.109 (heavy). Validation chose it for every segment, so the switching hybrid became popularity.
  - **ALS ties popularity on active users** (0.118; paired difference −0.005 to +0.023) and loses on light users (0.099). It can't serve new users at all.
  - **Knowledge-based:** about 0.055. It is capped by a catalogue that covers only 14% of targets, and it trails popularity by 0.077 on new users. It does offer explanations and more novel suggestions (5.5–6.5 vs 3.4 bits).
  - **BioCLIP text embeddings beat taxonomy features** for content-based recommendations: +0.017 (paired range +0.011 to +0.024).
  - No model wins on both accuracy and novelty. [comparison_by_segment.png](results/recsys/comparison_by_segment.png), [accuracy_vs_novelty.png](results/recsys/accuracy_vs_novelty.png)
- **GBIF reconciliation** ([reconciliation.json](results/kg/reconciliation.json)):
  - Data added sambar and greater mouse-deer (SG) and bearded pig (MY).
  - Red muntjac (SG) is kept but flagged unconfirmed (open world).
  - **Threatened species' coordinates are blurred:** > 10 km uncertainty for 79% of CR, 95% of EN and 7% of LC records.
  - The precision guard blocked 25 zone links and allowed 31. Only one CR species (straw-headed bulbul) can be placed in any zone.
- **IUCN via GBIF:** 25/25 species matched exactly. The Malayan tiger subspecies matched only at a higher rank, so GBIF says EN; a cited override sets it to CR. In the KG, 3 taxa are CR, 6 EN and 9 VU.
- **Seed presence flags vs GBIF (Singapore):** records since 2015, capped at 600 per region:
  - Sambar: seeded absent, 94 records.
  - Greater mouse-deer: seeded absent, 42 records.
  - Red muntjac: seeded present, 0 records under that name.
  - Great argus: 6 SG records, none since 2015.
- **NLLB smoke run** (3 alerts × ms/zh/ta; these are qualitative examples, too few for rates) ([outputs.csv](results/translation/outputs.csv)):
  - **zh:** all 3 outputs stop at the first comma. The CR species and the "rangers respond" instruction are dropped.
  - **ta:** "Malayan tiger" becomes மலையாள புலி. The back-translation is "*Malayalam* tiger".
  - **ms, ta:** "critically endangered" is weakened to "endangered" (ms) or "dangerous" (ta).
  - **ms:** the raw output gives "harimau Melayu" and "pangolin Sunda"; glossary mode corrects these to *harimau Malaya* and *tenggiling*.
- **GBIF observer data** (iNaturalist research-grade, mammals + birds, SG + Peninsular MY):
  - 63,003 usable records from 6,618 observers, covering 728 species.
  - 35,213 observer–species pairs, a matrix density of **0.73%**.
  - **3,117 observers (47%) recorded only one species**; 1,353 recorded 5 or more.
  - Coverage: all SG and PenMY mammal records were downloaded (14,866 and 11,091). Birds were sampled at 20,000 of 137,518 (SG) and 19,996 of 53,394 (PenMY), allocated across years in proportion.
  - This is the cold-start problem in numbers, and the case for a switching hybrid.
- **Property graph** (all years; [README](src/reasoning/graph/README.md)):
  - The precision guard dropped **28%** of records (18,268), because their location is blurred by more than half a square (5.5 km).
  - 46,715 records kept: 580 species, 548 squares, 7,477 species–square links, 1,128 nodes, 9,863 edges.
  - **Survey effort is extremely concentrated: 89% of records come from 10% of squares.** The median square holds 4 species, and 159 squares hold only one. The species count of a square correlates 0.87 with its (log) record count, so "species-rich" here mostly means "well surveyed".
  - 25,981 species pairs share 3 or more squares. The top non-threatened pairs are in [top_cooccurrence.csv](results/graph/top_cooccurrence.csv).
- **Link prediction, preliminary** (validation only: train ≤ 2023, predict the 887 links first recorded in 2024; untuned single runs, not final):
  - Hits@10: popularity 0.17, spatial neighbour 0.20, co-occurrence (k=25) 0.24. The best TransE run reached 0.26.
  - So far the embeddings are roughly level with the co-occurrence baseline. Final numbers will come from the 2025–26 test with 5 seeds.
- **Knowledge-based recommender** (Sungei Buloh, threatened-first) → smooth-coated otter, straw-headed bulbul, long-tailed macaque. Each comes with reasons. No precision@k yet, because evaluation needs the observer data.

## Known issues / failures found

1. **There are no human camera-trap images.** LILA withholds them for privacy, and all 150 sampled human images returned 404. Person detection, the intrusion signal, cannot be evaluated on public camera-trap data.
2. **The WCS tigers are Sumatran, not Malayan** (302/321 from Indonesia). No public Malayan tiger camera-trap set was found.
3. **Most leopards are Kenyan** (382/470). The location split left only 3 Lao leopards in the test set, so the planned accuracy-based domain-gap test is inconclusive. A dedicated split (train on African leopards, test on Lao) is needed.
4. **The Sunda pangolin has only 17 images,** so it is zero-shot only.
5. **GBIF does not resolve the Malayan tiger subspecies.** Without the override, the main CR rule would miss the flagship species.
6. **Several hand-curated presence flags are wrong or unconfirmed** (see results above).
7. **Zone-ranking bias:** the habitat score is normalised by a zone's number of habitats, which favours zones with few habitats.
8. **GBIF's API stalls on deep paging.** Pages past offset ~10,000 send a partial response and then hang (measured: 40 s timeout, ~147 KB received). The download hung twice before being fixed by splitting each query by year. Bird records are still a capped sample, not a random one.
9. **NLLB truncates long alerts in Chinese,** which also breaks glossary placeholders (0/3 survived). Planned fix: split sentences before translating.
10. **NLLB loses the IUCN severity term and confuses Malayan with Malayalam in Tamil.** This is a safety-relevant failure for an alert system.
11. **My look-alike hypothesis was mostly wrong.** Zero-shot's errors come from text-prompt "attractor" classes (wild boar, tapir, tiger), not from visually similar pairs.
12. **Closed-set probes can't flag unknown species:** 8 of 17 pangolins were confidently called "clouded leopard". Zero-shot with an extra label handles this better (9/17 correct).
13. **kNN picked k=20, the edge of the search grid.** The grid should be extended.
14. **MegaDetector puts false "animal" boxes on 12% of empty frames** (6/50) at threshold 0.2.
15. ~~Review blind spot~~ **Fixed by rule C4** (see results). It costs about 2–4 extra percentage points of review workload, and it doesn't help zero-shot, whose mistakes are confident.
16. **Zone-level threat data doesn't discriminate.** Every zone has a VU/EN species in the records, so rule A4 makes every person alert at least MEDIUM.
17. **CR locations are mostly blurred in GBIF,** so the "CR species recorded in this zone" rule (A2) can only use the straw-headed bulbul. The design leans on camera evidence.
18. **A SPARQL scoping bug** (`FILTER` inside a `UNION` branch) silently disabled the camera part of rules A3 and A4. Scenario tests caught it, and it's fixed.
19. **Zero-shot confidence is uncalibrated** (softmax temperature 100, my choice). Its review numbers depend on that choice.
20. **The brief's switching hybrid didn't help.** Without a crossover between models, switching just picks the best single model (here, popularity). The "knowledge for cold start, learning for active users" story didn't hold for accuracy in this data.
21. **ALS tuning hit the edge of the grid** (regularisation 100). The best ALS behaves almost like popularity.
22. **Trail suggestions are unevaluated.** There's no ground truth for zone visits, and threatened-species coordinates are blurred.
23. **The "recorded next" target favours common species** by design, which helps popularity.
24. **Link prediction can't score never-seen species or squares.** On validation, 48 of 1,082 new links involve a species not yet in the graph, and 147 a square not yet in it. These are counted, not hidden.
25. **The precision guard removes most threatened-species records** (they are blurred), so the occurrence graph under-represents exactly the species the alerts care about.
26. **The TransE epoch grid is too short:** 9 of 24 runs were still improving at 150 epochs. Extend it before resuming.

## Decisions made

- 2026-11-15:
  - Camera-trap experiment: 9 WCS classes (SE Asia; leopards from all countries), up to 150 images per class, BioCLIP v1 as the primary model.
  - **Person detection:** evaluated on non-camera-trap public images of people, such as COCO. The domain gap (daylight, framed photos, no IR) will be stated explicitly.
  - **Recommender items:** species, across the shared 728-species catalogue. Trails are derived from species scores.
  - **Link prediction:** target = species–square links first recorded in 2025–26 (train ≤ 2024; tuning on 2024). TransE vs RotatE. Three non-embedding baselines (popularity, co-occurrence neighbours, spatial neighbour). The curated `livesIn` links (72) are too few to test on.
  - **Privacy:** threatened species (VU/EN/CR) are left out of every per-square output.

## Blocked on me

- [ ] **Reference translations:** fill `ref_ms` / `ref_ta` / `ref_zh` in `src/perception/translation/eval_set.csv`, ideally checked by native speakers. Until then, only back-translation and name checks are available.
- [ ] **Verify the multilingual glossary:** Malay and Chinese species names in `species.csv`. Tamil names are blank.

## Next up

1. **Resume link prediction:** extend `EPOCHS` in `src/reasoning/graph/experiment.py` to 300 (see issue 26; the cached runs then need re-running only for the longer epochs, or clear `data/cache/graph_val_grid.json`), then run `python -m src.reasoning.graph.experiment` (~45 min on CPU). After that: charts (comparison by square density, ablation, example map of an LC species), README results, portfolio section.
2. Optional recommender follow-up: a weighted hybrid (ALS or BioCLIP-content blended with popularity, weight tuned on validation).
3. MegaDetector person recall and false positives on a small COCO person/no-person set (domain gap stated).
4. Small camera-trap follow-ups: an Africa→Laos leopard split, and a wider kNN grid.
