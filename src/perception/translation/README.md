# Multilingual alerts with NLLB-200

## What it does
Translates sighting and intrusion alerts from English into **Malay (zsm_Latn)**, **Tamil (tam_Taml)** and **Simplified Chinese (zho_Hans)** with `facebook/nllb-200-distilled-600M`.

There are two modes (`nllb.py`):
- **raw:** translate the whole alert.
- **glossary:** species and place names are swapped for placeholders (`X1`, `X2`) before translating, then the ontology's multilingual labels are inserted (for example *Malayan tapir → tenuk*). If the model damages a placeholder, the output falls back to raw and the failure is counted.

## Files
- `templates.py`: 5 alert templates and a generator for the 20-alert evaluation set.
- `eval_set.csv`: English sources, with **blank reference columns for a native speaker to fill in**.
- `evaluate.py`: computes the following metrics, written to `results/translation/`:
  - chrF vs human references, where filled
  - back-translation chrF
  - species-name hit rate
  - place-name preservation
  - placeholder survival

## Concepts demonstrated
- Sequence-to-sequence models and multilingual MT.
- Evaluation without and with references. Back-translation is a *weak* signal: it rewards literal output and can hide errors that survive the round trip.

## Result
*Pending.*

## Failure modes to watch
- Species names translated literally or dropped.
- Place names ("Bukit Timah", "Taman Negara") altered.
- Placeholders mangled.
- Tamil script quality, which I cannot judge without a native speaker.

## Next
Fill the reference translations, then run the full evaluation for all 3 languages × 20 alerts.
