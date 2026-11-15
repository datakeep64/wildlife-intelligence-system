# Data

Everything in this folder is gitignored except the `download_*.py` scripts and this README.

| Script | Source | Output | Size |
|---|---|---|---|
| `download_iucn.py` | GBIF species API (IUCN Red List mirror) | `iucn_status.csv` | tiny |
| `download_lila.py` | LILA BC WCS Camera Traps | `lila_wcs/images/`, `lila_wcs/manifest.csv` | ~360 MB |
| `download_gbif.py` | GBIF occurrence search API | `gbif/seed_species_occurrences.csv`, `gbif/observers_occurrences.csv` | ~20-40 MB |

## Known data caveats (measured)

- **The WCS tigers are Sumatran, not Malayan.** Of the single-label tiger images, 302 of 321 come from Indonesia. The classifier learns "tiger", and the subspecies distinction is invisible to it.
- **Most leopards are Kenyan.** 382 of 470 leopard annotations are from Kenya. This is a deliberate domain-gap test: does a Kenyan savanna leopard look like a Lao forest leopard to BioCLIP?
- **There are no human images.** LILA withholds camera-trap images containing people for privacy. All 150 sampled human images returned HTTP 404, so person detection cannot be evaluated on this data.
- **The Sunda pangolin has only 17 images,** which is too few to train on. It is evaluated zero-shot only.
- **The GBIF sample is capped, not random.** The search API stalls on pages past offset ~10,000 (measured), so each query is split by year and the record budget is allocated across years in proportion. All SG and PenMY mammal records were downloaded. Birds are a 20k sample per region (out of 137,518 in SG and 53,394 in PenMY). Use the GBIF download API (free account) for complete data.
- **GBIF misses the Malayan tiger's subspecies status.** It resolves *Panthera tigris jacksoni* to the species, which is EN, while the subspecies is CR. `species.csv` carries a cited override.
