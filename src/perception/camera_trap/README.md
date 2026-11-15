# Camera-trap triage + species classification (PRS)

## What it does
`image → MegaDetector v5a → crop → BioCLIP embedding → species → sighting record`

1. **Triage** (`detect.py`). MegaDetector labels each frame animal, person, vehicle or empty. Empty frames are dropped, and person or vehicle frames become an intrusion signal.
2. **Crop** (`detect.py`). The top animal box is cropped as a padded square, so the classifier sees the animal rather than the scene.
3. **Embed** (`embed.py`). A frozen BioCLIP (ViT-B/16) produces L2-normalised image and text embeddings.
4. **Classify** (`classify.py`). The same frozen features feed three classifiers:
   - zero-shot (nearest taxonomic text prompt)
   - linear probe (logistic regression)
   - kNN
5. **Sighting** (`pipeline.py`). Each frame becomes a JSON record whose fields map to `wis:Sighting` in the ontology. This is the hand-off to the reasoning half.

## Course concepts (PRS)
- **Transfer learning and representations:** a foundation model trained on the Tree of Life is reused with no fine-tuning.
- **Zero-shot vs supervised:** does a small labelled set (about 110 images per class) beat text prompts on the *same* features? A gain from the probe means the representation separates the classes, but the prompts point to the wrong places.
- **Honest evaluation:**
  - The train/test split is grouped by camera location (`dataset.py`), so no camera appears in both.
  - Only one frame per burst sequence is kept.
  - Reporting covers per-class precision/recall and look-alike confusion (tiger↔leopard↔clouded leopard, sun bear↔wild boar, sambar↔muntjac), not only accuracy.

## Data
LILA WCS Camera Traps, Indonesia and Laos: 9 classes × up to 150 images, location-grouped 75/25 split. See [data/README.md](../../../data/README.md) for the caveats (Sumatran tigers, Kenyan leopards, no human images).

## Result
*Pending (Phase B).* The smoke test on 3 real images ran end to end (tiger, sun bear and empty frames handled correctly). That is a plumbing check, not a result.

## Failure modes to watch
- MegaDetector misses (small or occluded animals, night IR frames).
- Look-alike confusions, and the Kenyan→Lao leopard domain gap.
- Pangolin (17 images) is evaluated zero-shot only.
- The tiger subspecies is invisible to the model.

## Next
Run `experiment.py` (Phase B). It will produce the metrics JSON, confusion matrices, a PCA/UMAP embedding plot (linking to the Word Embedding Explorer), and a data-efficiency curve (probe accuracy at 1, 5, 10 and all shots per class).
