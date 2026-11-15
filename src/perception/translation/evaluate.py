"""Evaluate NLLB alert translation (en -> ms, ta, zh).

Metrics:
  chrF (vs human reference)  only for rows where ref_<lang> is filled in eval_set.csv
  back-translation chrF      en -> X -> en vs the original; needs no reference, but
                             rewards literal translations and can hide errors that
                             'round-trip' cleanly. Reported as a weak signal only.
  species-name hit rate      does the output contain the KG glossary name for the species?
  place-name preserved       does the official place name survive verbatim?
  placeholder survival       glossary mode: did the model keep X1/X2 tokens intact?

Usage: .venv/bin/python -m src.perception.translation.evaluate [--langs ms zh ta] [--limit 5]
Output: results/translation/{outputs.csv, metrics.json}
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sacrebleu.metrics import CHRF

from src.common.paths import RESULTS
from src.perception.translation.nllb import get_translator, translate_alert
from src.perception.translation.templates import glossary_for, names

EVAL_CSV = Path(__file__).with_name("eval_set.csv")
OUT = RESULTS / "translation"
chrf = CHRF()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["ms", "zh", "ta"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ev = pd.read_csv(EVAL_CSV, dtype=str).fillna("")
    if args.limit:
        ev = ev.head(args.limit)
    sp, zn = names()
    tr = get_translator()
    rows = []
    for lang in args.langs:
        for _, r in ev.iterrows():
            g = glossary_for(r.species_key or None, r.zone_key, lang)
            res = translate_alert(r.en, lang, glossary=g, translator=tr)
            back = tr.translate([res["raw"]], "en", src=lang)[0]
            sp_name = sp.loc[r.species_key, f"name_{lang}"] if r.species_key else ""
            place = zn.loc[r.zone_key, "name"]
            rows.append({
                "id": r.id, "lang": lang, "en": r.en, "raw": res["raw"], "glossary": res["glossary"],
                "back_translation": back, "reference": r[f"ref_{lang}"],
                "placeholders_ok": res["placeholders_ok"],
                "species_glossary_name": sp_name,
                "raw_has_species_name": (sp_name.lower() in res["raw"].lower()) if sp_name else None,
                "raw_place_preserved": place.lower() in res["raw"].lower(),
                "backtrans_chrf": round(chrf.sentence_score(back, [r.en]).score, 2),
            })
            print(f"[{lang}] {r.id} raw: {res['raw']}")
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "outputs.csv", index=False)

    metrics = {}
    for lang, d in df.groupby("lang"):
        m = {
            "n": int(len(d)),
            "backtrans_chrf_mean": round(float(d.backtrans_chrf.mean()), 2),
            "raw_species_name_hit_rate": _rate(d.raw_has_species_name),
            "raw_place_preserved_rate": _rate(d.raw_place_preserved),
            "glossary_placeholder_survival": _rate(d.placeholders_ok),
        }
        with_ref = d[d.reference != ""]
        if len(with_ref):
            m["chrf_raw_vs_ref"] = round(chrf.corpus_score(list(with_ref.raw), [list(with_ref.reference)]).score, 2)
            m["chrf_glossary_vs_ref"] = round(chrf.corpus_score(list(with_ref.glossary), [list(with_ref.reference)]).score, 2)
            m["n_with_reference"] = int(len(with_ref))
        else:
            m["chrf_vs_ref"] = "no human references yet"
        metrics[lang] = m
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def _rate(s: pd.Series):
    s = s.dropna()
    return round(float(s.astype(bool).mean()), 3) if len(s) else None


if __name__ == "__main__":
    main()
