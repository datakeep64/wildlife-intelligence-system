# Recommenders

The architecture is chosen per scenario, and all are scored by the same `evaluate()` (precision@k, recall@k, hit-rate@k, coverage) on the same split.

| Scenario | Architecture | Why | Status |
|---|---|---|---|
| New visitor, no history | **Knowledge/constraint-based** on the ontology KG | Cold start: there is nothing to learn from, but plenty we know | Built |
| Active iNaturalist observer | Collaborative filtering (implicit ALS, observer × species) | Implicit feedback is plentiful | Phase B |
| "Species like this one" | Content-based (KG attributes / BioCLIP embeddings) | Rare species have no interactions | Phase B |
| Production | Switching hybrid | Picks a model per user regime | Phase B |
| Baseline | Popularity | What every model must beat | Built |

## Files
- `base.py`:
  - `Recommender` interface (`fit`, `recommend`, `evaluate`)
  - `Rec` (item, score, **reasons**)
  - `PopularityRecommender`
  - `leave_last_out_split`
- `knowledge_based.py`:
  - **Hard constraints:** country, zone habitat, IUCN floor.
  - **Soft preferences:** habitat interests, preference for threatened species.
  - **Explanations:** every recommendation lists its reasons.
  - **Trail suggestions:** `recommend_zones()` ranks zones by the score of the species they could hold.

## Example (real output)
```
Visitor: {country: SG, zone: sungei_buloh, prefer_threatened: True}
  Smooth-coated otter   1.25  <- lives in mangrove, wetland, found in Sungei Buloh; IUCN VU
  Straw-headed bulbul   1.00  <- lives in wetland, found in Sungei Buloh; IUCN CR
  Long-tailed macaque   0.88  <- lives in mangrove, found in Sungei Buloh; IUCN EN
```

## Known issues
- **Zone-score bias:** the habitat score is normalised by a zone's number of habitats. Bukit Timah (2 habitats, 8 species) therefore outranks Central Catchment (3 habitats, 9 species).
- **Different item catalogues:** the knowledge-based recommender covers only the 26 KG taxa. *Decision (2026-11-15):* every model ranks species over the shared 728-species iNaturalist catalogue, and trails are derived from species scores. Consequence: the knowledge-based recommender can only ever hit the 26 curated taxa, which caps its recall. That cap is itself a finding about curated knowledge.
