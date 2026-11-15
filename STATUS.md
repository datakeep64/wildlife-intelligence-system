# STATUS

_Last updated: 2026-11-15 (session 1, Phase A)_

## Components

| Component | Status | What exists |
|---|---|---|
| Repo scaffold | Done | Folders, `requirements.txt` + lockfile (Python 3.12 venv), `.gitignore`, README with Mermaid architecture |
| Data: IUCN status | Done | `data/download_iucn.py` → statuses for 26 taxa via GBIF |
| Data: LILA camera traps | Done | `data/download_lila.py` → 1,400 WCS images, location-grouped split |
| Data: GBIF occurrences | Done | `data/download_gbif.py`: 7,062 seed-species records + 65,953 iNaturalist observer records (year-stratified sample) |
| Ontology KG (schema + seed) | Working (results pending) | `ontology.ttl` (TBox), `species_seed.ttl` (ABox, 435 triples); SPARQL works. Alert rules not written yet |
| Camera-trap pipeline | Working (results pending) | MegaDetector → crop → BioCLIP → zero-shot/LR/kNN code; smoke test passes on real images. Full experiment not run |
| NLLB translation | Working (results pending) | Module (raw + glossary modes), 20-alert eval set, evaluator; smoke run on 3 alerts × 3 languages done ([outputs](results/translation/outputs.csv)) |
| Recommender framework | Working (results pending) | Shared `fit/recommend/evaluate`, popularity baseline, knowledge-based recommender with explanations + trail suggestions |
| Property graph | Not started | Phase B |
| Embedding KG (PyKEEN) | Not started | Phase B (pykeen installed) |
| CF (ALS) / content-based / hybrid | Not started | Phase B (implicit installed) |
| Bioacoustics (optional) | Not started | |
| Integration loop | Not started | Phase C |
| Portfolio demo page | Not started | Phase C |

## Results so far (real, measured)

- **Camera-trap dataset** ([data/README.md](data/README.md)):
  - 9 classes × 133–150 images, 1 frame per burst sequence, plus 17 pangolin images (zero-shot only) and 50 empty frames.
  - The test split is grouped by camera location and holds 25% of each class.
- **Smoke test** (3 images; a plumbing check, not an accuracy claim):
  - tiger crop → `tiger` p=0.998
  - sun bear → `sun_bear` p=0.978
  - empty frame → MegaDetector `empty`
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
- **Knowledge-based recommender** (Sungei Buloh, threatened-first) → smooth-coated otter, straw-headed bulbul, long-tailed macaque. Each comes with reasons. No precision@k yet, because evaluation needs the observer data.

## Known issues / failures found

1. **There are no human camera-trap images.** LILA withholds them for privacy, and all 150 sampled human images returned 404. Person detection, the intrusion signal, cannot be evaluated on public camera-trap data.
2. **The WCS tigers are Sumatran, not Malayan** (302/321 from Indonesia). No public Malayan tiger camera-trap set was found.
3. **Most leopards are Kenyan** (382/470). This will be reported as a domain-gap analysis.
4. **The Sunda pangolin has only 17 images,** so it is zero-shot only.
5. **GBIF does not resolve the Malayan tiger subspecies.** Without the override, the main CR rule would miss the flagship species.
6. **Several hand-curated presence flags are wrong or unconfirmed** (see results above).
7. **Zone-ranking bias:** the habitat score is normalised by a zone's number of habitats, which favours zones with few habitats.
8. **GBIF's API stalls on deep paging.** Pages past offset ~10,000 send a partial response and then hang (measured: 40 s timeout, ~147 KB received). The download hung twice before being fixed by splitting each query by year. Bird records are still a capped sample, not a random one.
9. **NLLB truncates long alerts in Chinese,** which also breaks glossary placeholders (0/3 survived). Planned fix: split sentences before translating.
10. **NLLB loses the IUCN severity term and confuses Malayan with Malayalam in Tamil.** This is a safety-relevant failure for an alert system.

## Decisions made

- 2026-11-15:
  - Camera-trap experiment: 9 WCS classes (SE Asia; leopards from all countries), up to 150 images per class, BioCLIP v1 as the primary model.
  - **Person detection:** evaluated on non-camera-trap public images of people, such as COCO. The domain gap (daylight, framed photos, no IR) will be stated explicitly.
  - **Recommender items:** species, across the shared 728-species catalogue. Trails are derived from species scores.

## Blocked on me

- [ ] **Reference translations:** fill `ref_ms` / `ref_ta` / `ref_zh` in `src/perception/translation/eval_set.csv`, ideally checked by native speakers. Until then, only back-translation and name checks are available.
- [ ] **Verify the multilingual glossary:** Malay and Chinese species names in `species.csv`. Tamil names are blank.

## Next up

1. Run the camera-trap experiment (zero-shot vs LR vs kNN, confusion matrices, look-alike analysis, UMAP, few-shot curve) → `results/camera_trap/`.
2. Write the SPARQL alert rules, plus a GBIF reconciliation rule for presence → `results/kg/`.
3. ALS and content-based recommenders, compared against popularity and knowledge-based on precision@k / recall@k over the 728 species.
4. MegaDetector person recall and false positives on a small COCO person/no-person set (domain gap stated).
