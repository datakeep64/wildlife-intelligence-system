"""Reconcile hand-curated facts with GBIF occurrence data -> observations.ttl.

Two explicit, documented rules (instead of silently editing species.csv):

1. Country presence (wis:occursInCountry)
   assert if   seed says 'Y'   OR   >= MIN_COUNTRY_RECORDS recent records in that country
   'recent' = year >= 2015 and a human/machine observation (not a museum specimen).
   Open-world assumption: zero records is NOT evidence of absence, so a seeded 'Y'
   is never retracted - it is only flagged as unconfirmed.

2. Zone occurrence (wis:observedIn)
   a record belongs to a zone if it lies within the zone's radius (SG 3 km, MY 20 km)
   AND its stated coordinate uncertainty fits inside that radius (precision guard).
   assert if >= MIN_ZONE_RECORDS such recent records.
   Why the guard: iNaturalist/GBIF deliberately blur threatened species' locations
   (typically ~31 km) to protect them from poachers. A blurred record cannot say which
   reserve the animal was in, so it supports country presence only.

Outputs: src/reasoning/kg/observations.ttl, results/kg/{presence_reconciliation.csv,
         zone_observations.csv, reconciliation.json}
Usage:   .venv/bin/python -m src.reasoning.kg.reconcile
"""
import json

import numpy as np
import pandas as pd
from rdflib import Graph, Literal, Namespace

from src.common.paths import GBIF_DIR, IUCN_CSV, KG_DIR, RESULTS, SPECIES_CSV, ZONES_CSV

WIS = Namespace("https://datakeep64.github.io/wis/ontology#")
OBS_TTL = KG_DIR / "observations.ttl"
OUT = RESULTS / "kg"
MIN_COUNTRY_RECORDS = 3
MIN_ZONE_RECORDS = 2
RADIUS_KM = {"SG": 3.0, "MY": 20.0}
REGION_TO_COUNTRY = {"SG": "SG", "PenMY": "MY"}


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def load_recent() -> pd.DataFrame:
    o = pd.read_csv(GBIF_DIR / "seed_species_occurrences.csv", low_memory=False)
    o = o[(o.year >= 2015) & o.basisOfRecord.isin(["HUMAN_OBSERVATION", "MACHINE_OBSERVATION"])].copy()
    o["country"] = o.region.map(REGION_TO_COUNTRY)
    o["unc_km"] = o.coordinateUncertaintyInMeters / 1000
    return o


def presence(o: pd.DataFrame, species: pd.DataFrame) -> pd.DataFrame:
    counts = o.groupby(["kg_key", "country"]).size()
    rows = []
    for _, s in species.iterrows():
        for country, col in (("SG", "occurs_sg_seed"), ("MY", "occurs_penmy_seed")):
            n = int(counts.get((s.key, country), 0))
            seed = s[col]
            data_says = n >= MIN_COUNTRY_RECORDS
            assert_it = seed == "Y" or data_says
            if seed == "Y" and n == 0:
                status = "seed only (unconfirmed by data)"
            elif seed == "Y" and data_says:
                status = "seed confirmed"
            elif seed == "Y":
                status = "seed, weak data support"
            elif data_says:
                status = f"ADDED by data (seed was '{seed}')"
            elif n > 0:
                status = f"below threshold ({n} records), not asserted"
            else:
                status = "absent in seed and data"
            rows.append({"key": s.key, "country": country, "seed": seed, "recent_records": n,
                         "asserted": assert_it, "status": status})
    return pd.DataFrame(rows)


def zone_links(o: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, z in zones.iterrows():
        r = RADIUS_KM[z.country]
        sub = o[o.country == z.country].copy()
        sub["dist_km"] = haversine_km(sub.decimalLatitude, sub.decimalLongitude, float(z.lat), float(z.lon))
        inside = sub[sub.dist_km <= r]
        precise = inside[inside.unc_km.fillna(0) <= r]  # missing uncertainty treated as precise (stated in README)
        for key in sorted(set(inside.kg_key)):
            n_in, n_prec = int((inside.kg_key == key).sum()), int((precise.kg_key == key).sum())
            rows.append({"zone": z.key, "key": key, "records_in_radius": n_in, "precise_records": n_prec,
                         "asserted": n_prec >= MIN_ZONE_RECORDS,
                         "blocked_by_precision_guard": n_in >= MIN_ZONE_RECORDS and n_prec < MIN_ZONE_RECORDS})
    return pd.DataFrame(rows)


def main() -> None:
    species = pd.read_csv(SPECIES_CSV, dtype=str).fillna("")
    species = species[species["rank"] == "species"]  # GBIF records are species-level; subspecies via rules
    zones = pd.read_csv(ZONES_CSV, dtype=str)
    iucn = pd.read_csv(IUCN_CSV, dtype=str).set_index("key")["final_code"]
    o = load_recent()

    pres = presence(o, species)
    links = zone_links(o, zones)

    g = Graph()
    g.bind("wis", WIS)
    for _, p in pres[pres.asserted].iterrows():
        g.add((WIS[p.key], WIS.occursInCountry, WIS[p.country]))
    for _, l in links[links.asserted].iterrows():
        g.add((WIS[l.key], WIS.observedIn, WIS[l.zone]))
    g.serialize(OBS_TTL, format="turtle")

    OUT.mkdir(parents=True, exist_ok=True)
    pres.to_csv(OUT / "presence_reconciliation.csv", index=False)
    links.to_csv(OUT / "zone_observations.csv", index=False)

    o["iucn"] = o.kg_key.map(iucn)
    blur = o.assign(blurred=o.unc_km > 10).groupby("iucn").blurred.mean().round(3).to_dict()
    asserted = links[links.asserted].assign(iucn=lambda d: d.key.map(iucn))
    summary = {
        "rules": {"min_country_records": MIN_COUNTRY_RECORDS, "min_zone_records": MIN_ZONE_RECORDS,
                  "radius_km": RADIUS_KM, "recent": "year >= 2015, human or machine observation"},
        "recent_records": int(len(o)),
        "share_records_blurred_over_10km_by_iucn": blur,
        "presence_status_counts": pres.status.value_counts().to_dict(),
        "presence_changes": pres[pres.status.str.startswith(("ADDED", "seed only"))][["key", "country", "seed", "recent_records", "status"]].to_dict("records"),
        "zone_links_asserted": int(links.asserted.sum()),
        "zone_links_blocked_by_precision_guard": int(links.blocked_by_precision_guard.sum()),
        "asserted_links_by_iucn": asserted.iucn.value_counts().to_dict(),
        "cr_zone_links": asserted[asserted.iucn == "CR"][["zone", "key"]].to_dict("records"),
        "observations_ttl_triples": len(g),
    }
    (OUT / "reconciliation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {OBS_TTL} ({len(g)} triples)")


if __name__ == "__main__":
    main()
