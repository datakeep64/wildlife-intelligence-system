"""Download GBIF occurrence records for Singapore + Peninsular Malaysia.

Two modes:
  observers     iNaturalist research-grade records (mammals + birds) with the
                observer name (`recordedBy`) -> observer x species matrix for
                collaborative filtering, and the property graph.
  seed_species  records for each taxon in species.csv (any dataset) -> verifies
                the hand-entered occurs_sg/occurs_penmy flags, and feeds
                wis:observedIn links.

Deep paging (measured 2026-11-15): pages up to offset ~10,000 return in ~2 s.
Beyond that the API sends a partial body and then stalls, so read timeouts never
fire. We therefore never page past MAX_OFFSET, and split each query by YEAR: one
facet call gets the per-year counts, and the --max-records budget is allocated
across years in proportion. Within a year, records come in GBIF's default order
(not random). That bias is smaller than "first N overall", but still state it in
the write-up. For complete data, use the GBIF download API (free account).

Peninsular Malaysia = country MY AND inside a bounding box west of 104.6E
(excludes Sabah/Sarawak on Borneo).

Output: data/gbif/{observers,seed_species}_occurrences.csv
Usage:  .venv/bin/python data/download_gbif.py [--mode both] [--max-records 20000]
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.paths import GBIF_DIR, IUCN_CSV, SPECIES_CSV  # noqa: E402

API = "https://api.gbif.org/v1/occurrence/search"
INAT_RG = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"  # iNaturalist research-grade dataset on GBIF
CLASSES = {"Mammalia": 359, "Aves": 212}
PEN_MY_WKT = "POLYGON((99.6 1.2,104.6 1.2,104.6 6.8,99.6 6.8,99.6 1.2))"
REGIONS = {
    "SG": {"country": "SG"},
    "PenMY": {"country": "MY", "geometry": PEN_MY_WKT},
}
FIELDS = [
    "gbifID", "datasetKey", "basisOfRecord", "species", "speciesKey", "taxonKey", "scientificName",
    "class", "order", "family", "recordedBy", "eventDate", "year", "month",
    "decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters", "stateProvince", "locality",
    "iucnRedListCategory",
]


MAX_OFFSET = 9_900  # deeper pages stall (see docstring)
BASE = {"hasCoordinate": "true", "hasGeospatialIssue": "false", "occurrenceStatus": "PRESENT"}


def get(params: dict) -> dict:
    for attempt in range(4):
        try:
            r = requests.get(API, params={**BASE, **params}, timeout=(10, 30))  # (connect, read)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"GBIF request failed 4x: {params}")


def page_through(params: dict, max_records: int, desc: str) -> list[dict]:
    rows, offset, limit = [], 0, 300
    with tqdm(total=max_records, desc=desc, leave=False) as bar:
        while offset < min(max_records, MAX_OFFSET + 1):
            d = get({**params, "limit": min(limit, max_records - offset), "offset": offset})
            rows += [{f: rec.get(f) for f in FIELDS} for rec in d["results"]]
            bar.update(len(d["results"]))
            if d.get("endOfRecords") or not d["results"]:
                break
            offset += limit
            time.sleep(0.1)
    return rows[:max_records]


def year_counts(params: dict) -> dict[int, int]:
    d = get({**params, "limit": 0, "facet": "year", "facetLimit": 200})
    facets = d.get("facets") or [{"counts": []}]
    return {int(c["name"]): int(c["count"]) for c in facets[0]["counts"]}


def allocate(counts: dict[int, int], budget: int) -> dict[int, int]:
    """Proportional per-year quota, capped by availability and by MAX_OFFSET."""
    total = sum(counts.values())
    if total <= budget:
        quota = dict(counts)
    else:
        quota = {y: round(budget * c / total) for y, c in counts.items()}
    return {y: min(q, counts[y], MAX_OFFSET + 300) for y, q in quota.items() if q > 0}


def page_by_year(params: dict, budget: int, desc: str) -> tuple[list[dict], dict]:
    counts = year_counts(params)
    quota = allocate(counts, budget)
    rows = []
    for y in sorted(quota, reverse=True):
        rows += page_through({**params, "year": y}, quota[y], f"{desc}/{y}")
    info = {"available": sum(counts.values()), "fetched": len(rows), "years": f"{min(quota, default='-')}-{max(quota, default='-')}"}
    return rows, info


def observers(max_records: int) -> pd.DataFrame:
    """Each region x class chunk is cached, so an interrupted run resumes where it stopped."""
    chunks = []
    for region, rp in REGIONS.items():
        for cname, ckey in CLASSES.items():
            chunk = GBIF_DIR / f"_chunk_observers_{region}_{cname}_{max_records}.csv"
            if chunk.exists():
                df = pd.read_csv(chunk, low_memory=False)
            else:
                rows, info = page_by_year({**rp, "datasetKey": INAT_RG, "taxonKey": ckey}, max_records, f"{region}/{cname}")
                df = pd.DataFrame([{**r, "region": region} for r in rows])
                df.to_csv(chunk, index=False)
                print(f"{region:6s} {cname:9s} available={info['available']} years={info['years']}", flush=True)
            print(f"{region:6s} {cname:9s} {len(df):6d} records", flush=True)
            chunks.append(df)
    return pd.concat(chunks, ignore_index=True)


def seed_species(max_per_taxon: int) -> pd.DataFrame:
    if not IUCN_CSV.exists():
        sys.exit(f"{IUCN_CSV} missing - run data/download_iucn.py first (it resolves GBIF taxon keys)")
    keys = pd.read_csv(IUCN_CSV, dtype=str).set_index("key")["gbif_usage_key"]
    species = pd.read_csv(SPECIES_CSV, dtype=str)
    out = []
    for _, s in species[species["rank"] == "species"].iterrows():
        for region, rp in REGIONS.items():
            rows = page_through({**rp, "taxonKey": keys[s.key]}, max_per_taxon, f"{s.key}/{region}")
            out += [{**r, "region": region, "kg_key": s.key} for r in rows]
            print(f"{s.key:22s} {region:6s} {len(rows):5d}")
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["observers", "seed_species", "both"], default="both")
    ap.add_argument("--max-records", type=int, default=20_000, help="per region x class (observers mode)")
    ap.add_argument("--max-per-taxon", type=int, default=600, help="per region x taxon (seed_species mode)")
    args = ap.parse_args()
    GBIF_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode in ("seed_species", "both"):
        df = seed_species(args.max_per_taxon)
        df.to_csv(GBIF_DIR / "seed_species_occurrences.csv", index=False)
        print(f"wrote seed_species_occurrences.csv ({len(df)} rows)")
    if args.mode in ("observers", "both"):
        df = observers(args.max_records)
        df.to_csv(GBIF_DIR / "observers_occurrences.csv", index=False)
        print(f"wrote observers_occurrences.csv ({len(df)} rows, {df.recordedBy.nunique()} observers, "
              f"{df.species.nunique()} species)")


if __name__ == "__main__":
    main()
