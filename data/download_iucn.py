"""Fetch IUCN Red List categories for every taxon in species.csv via the GBIF API.

Why GBIF and not the IUCN API? The IUCN API needs a personal token; GBIF mirrors
the Red List categories openly. Known limitation (measured, not assumed): GBIF
matches some subspecies (e.g. Panthera tigris jacksoni) only to species level,
so we record the match type and let species.csv carry a cited manual override.

Output: data/iucn_status.csv
Usage:  .venv/bin/python data/download_iucn.py
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.paths import IUCN_CSV, SPECIES_CSV  # noqa: E402

API = "https://api.gbif.org/v1"


def gbif_match(name: str) -> dict:
    r = requests.get(f"{API}/species/match", params={"name": name}, timeout=30)
    r.raise_for_status()
    return r.json()


def gbif_iucn(usage_key: int) -> dict:
    r = requests.get(f"{API}/species/{usage_key}/iucnRedListCategory", timeout=30)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    return r.json()


def main() -> None:
    species = pd.read_csv(SPECIES_CSV, dtype=str).fillna("")
    rows = []
    for _, s in species.iterrows():
        m = gbif_match(s.scientific_name)
        key = m.get("usageKey")
        iucn = gbif_iucn(key) if key else {}
        rows.append(
            {
                "key": s.key,
                "scientific_name": s.scientific_name,
                "gbif_usage_key": key,
                "gbif_match_rank": m.get("rank"),
                "gbif_match_type": m.get("matchType"),  # EXACT / FUZZY / HIGHERRANK / NONE
                "gbif_iucn_code": iucn.get("code", ""),
                "gbif_iucn_name": iucn.get("scientificName", ""),
                "override_code": s.iucn_override,
                "final_code": s.iucn_override or iucn.get("code", "NE"),
                "source": "override: " + s.iucn_override_source if s.iucn_override else "GBIF/IUCN",
            }
        )
        print(f"{s.key:22s} {m.get('matchType', ''):10s} GBIF={iucn.get('code', '-'):3s} final={rows[-1]['final_code']}")
        time.sleep(0.2)  # be polite to the public API
    IUCN_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(IUCN_CSV, index=False)
    print(f"wrote {IUCN_CSV}")


if __name__ == "__main__":
    main()
