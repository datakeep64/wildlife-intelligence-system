# Knowledge graphs (IRS)

Three KG types, each chosen for a different job:

| KG | Tech | Why this type | Status |
|---|---|---|---|
| **Ontology KG** | RDF/OWL, rdflib, SPARQL | Curated facts and rules need explicit semantics: classes, a defined class (`ThreatenedTaxon`), and status ranks that rules can compare. | Schema + seed built |
| **Property graph** | NetworkX | Occurrence data is large, messy and relational (who saw what, where, when). Graph analytics such as co-occurrence fit it better than OWL. | Phase B |
| **Embedding KG** | PyKEEN (TransE vs another model) | Predicts *missing* species–habitat links that nobody curated. Extends my TransE/UMLS assignment. | Phase B |

## Files
- `ontology.ttl`: **TBox** (schema). Classes `Taxon/Species/Subspecies`, `Habitat`, `ConservationStatus` (with ordinal `statusRank`), `Threat`, `Zone/ProtectedArea`, `Sighting` (Animal/Person/Vehicle), `Alert`; plus controlled vocabularies.
- `species.csv`, `zones.csv`: curated seed facts. 26 taxa (Malay and Chinese names; Tamil left blank for verification) and 9 SG/MY protected areas.
- `build_seed.py`: builds the **ABox** (`species_seed.ttl`) from the CSVs plus IUCN data. Seed facts I'm unsure of are marked `?` and are *not* asserted (open-world assumption), so GBIF data can confirm or reject them.
- `graph.py`: loads TBox + ABox and runs SPARQL helpers.

## Course concepts (IRS)
- Knowledge representation: TBox vs ABox, OWL defined classes, the open-world assumption.
- Rule-based reasoning (Phase B): *CR taxon observed in zone Z* AND *person detected in Z within T hours* → HIGH alert.
- Curated vs automatically collected knowledge (see the findings below).

## Findings so far (real)
- **IUCN status via GBIF:** all 25 species matched exactly. The Malayan tiger subspecies matched only at a higher rank, so GBIF reports EN. The cited CR override in `species.csv` prevents the main rule from missing the flagship species.
- **Seed vs GBIF presence (Singapore):** my hand-entered flags disagree with GBIF for several species:
  - Sambar: seeded absent, 94 SG records since 2015.
  - Greater mouse-deer: seeded absent, 42 records.
  - Red muntjac: seeded present, 0 records under that name.
  - Great argus: 6 SG records, none since 2015.

  Phase B will reconcile these with an explicit, documented rule.

## Next
Populate `observedIn` from GBIF, add SPARQL alert rules, build the property graph, then run PyKEEN link prediction on held-out `livesIn` triples.
