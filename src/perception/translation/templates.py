"""Alert templates + generator for the translation evaluation set.

Run: .venv/bin/python -m src.perception.translation.templates
-> writes src/perception/translation/eval_set.csv (references left blank for a
   native speaker to fill; chrF is only meaningful against human references).
"""
from pathlib import Path

import pandas as pd

from src.common.paths import SPECIES_CSV, ZONES_CSV

TEMPLATES = {
    "intrusion_cr": "High-priority alert: a person was detected by camera {camera} in {zone}, where the critically endangered {species} has been recorded. Rangers should respond immediately.",
    "sighting": "Camera {camera} in {zone} recorded {a_species} at {time}.",
    "visitor_advisory": "Trail advisory: {a_species} was seen near {zone} today. Keep a safe distance and do not feed wildlife.",
    "vehicle_night": "A vehicle was detected at night by camera {camera} in {zone}. Please check for illegal entry.",
    "road_crossing": "Drivers near {zone}, please slow down: the {species} crosses this road at night.",
}

EVAL_CASES = [
    # (template, species_key, zone_key, camera, time)
    ("intrusion_cr", "malayan_tiger", "taman_negara", "TN-07", None),
    ("intrusion_cr", "sunda_pangolin", "central_catchment", "CC-12", None),
    ("intrusion_cr", "malayan_tiger", "endau_rompin", "ER-03", None),
    ("intrusion_cr", "straw_headed_bulbul", "pulau_ubin", "PU-02", None),
    ("sighting", "malayan_tapir", "taman_negara", "TN-11", "02:14"),
    ("sighting", "sun_bear", "royal_belum", "RB-05", "23:40"),
    ("sighting", "leopard", "krau", "KR-01", "04:55"),
    ("sighting", "wild_boar", "bukit_timah", "BT-04", "19:20"),
    ("sighting", "clouded_leopard", "endau_rompin", "ER-09", "01:05"),
    ("visitor_advisory", "wild_boar", "bukit_timah", None, None),
    ("visitor_advisory", "smooth_otter", "sungei_buloh", None, None),
    ("visitor_advisory", "longtailed_macaque", "central_catchment", None, None),
    ("visitor_advisory", "asian_elephant", "royal_belum", None, None),
    ("visitor_advisory", "sunda_colugo", "bukit_timah", None, None),
    ("vehicle_night", None, "taman_negara", "TN-02", None),
    ("vehicle_night", None, "krau", "KR-06", None),
    ("road_crossing", "malayan_tapir", "kuala_selangor", None, None),
    ("road_crossing", "sunda_pangolin", "central_catchment", None, None),
    ("road_crossing", "leopard_cat", "pulau_ubin", None, None),
    ("road_crossing", "malayan_tapir", "endau_rompin", None, None),
]


def names() -> tuple[pd.DataFrame, pd.DataFrame]:
    sp = pd.read_csv(SPECIES_CSV, dtype=str).fillna("").set_index("key")
    zn = pd.read_csv(ZONES_CSV, dtype=str).fillna("").set_index("key")
    return sp, zn


def render(template: str, species_key: str | None, zone_key: str, camera: str | None, time: str | None) -> str:
    sp, zn = names()
    species = sp.loc[species_key, "common_en"] if species_key else ""
    # lower-case common names mid-sentence except proper-noun parts (Malayan, Sunda, Asian)
    if species and species.split()[0] not in {"Malayan", "Sunda", "Asian", "Southern"}:
        species = species[0].lower() + species[1:]
    a_species = ("an " if species[:1].lower() in "aeiou" else "a ") + species if species else ""
    return TEMPLATES[template].format(species=species, a_species=a_species, zone=zn.loc[zone_key, "name"], camera=camera, time=time)


def glossary_for(species_key: str | None, zone_key: str, lang: str) -> dict[str, str]:
    """English term -> target term, from the ontology labels. Place names stay as-is
    (official names); species use the KG label for that language if we have one."""
    sp, zn = names()
    g = {zn.loc[zone_key, "name"]: zn.loc[zone_key, "name"]}
    if species_key:
        target = sp.loc[species_key, f"name_{lang}"]
        en_name = sp.loc[species_key, "common_en"]
        if target:
            g[en_name] = target
            g[en_name[0].lower() + en_name[1:]] = target
    return g


def make_eval_set() -> pd.DataFrame:
    rows = []
    for i, (t, s, z, cam, time) in enumerate(EVAL_CASES, start=1):
        rows.append({
            "id": f"A{i:02d}", "template": t, "species_key": s or "", "zone_key": z,
            "en": render(t, s, z, cam, time),
            "ref_ms": "", "ref_ta": "", "ref_zh": "",
            "ref_checked_by": "",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = Path(__file__).with_name("eval_set.csv")
    df = make_eval_set()
    df.to_csv(out, index=False)
    print(df[["id", "en"]].to_string(index=False))
    print(f"\nwrote {out}")
