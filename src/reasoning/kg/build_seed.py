"""Build the ABox (instance data) from curated CSVs + IUCN data -> species_seed.ttl.

TBox (ontology.ttl) = the schema. ABox (this output) = facts about species and zones.
Keeping them separate means Phase B can regenerate facts from GBIF without
touching the schema.

Usage: .venv/bin/python -m src.reasoning.kg.build_seed
"""
import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, RDFS, XSD

from src.common.paths import IUCN_CSV, SEED_TTL, SPECIES_CSV, ZONES_CSV

WIS = Namespace("https://datakeep64.github.io/wis/ontology#")


def build() -> Graph:
    g = Graph()
    g.bind("wis", WIS)
    sp = pd.read_csv(SPECIES_CSV, dtype=str).fillna("")
    zones = pd.read_csv(ZONES_CSV, dtype=str).fillna("")

    if IUCN_CSV.exists():
        iucn = pd.read_csv(IUCN_CSV, dtype=str).fillna("").set_index("key")
    else:
        print(f"WARNING: {IUCN_CSV} missing - run data/download_iucn.py. Using overrides only.")
        iucn = pd.DataFrame()

    for _, s in sp.iterrows():
        t = WIS[s.key]
        g.add((t, RDF.type, WIS.Subspecies if s["rank"] == "subspecies" else WIS.Species))
        g.add((t, WIS.scientificName, Literal(s.scientific_name)))
        for lang, col in [("en", "common_en"), ("ms", "name_ms"), ("zh", "name_zh"), ("ta", "name_ta")]:
            if s[col]:
                g.add((t, RDFS.label, Literal(s[col], lang=lang)))
        if s.parent:
            g.add((t, WIS.subspeciesOf, WIS[s.parent]))
        if s.wcs_label:
            g.add((t, WIS.wcsLabel, Literal(s.wcs_label)))

        # IUCN status: cited override > GBIF > Not Evaluated
        if s.key in getattr(iucn, "index", []):
            code, source = iucn.loc[s.key, "final_code"], iucn.loc[s.key, "source"]
        elif s.iucn_override:
            code, source = s.iucn_override, "override: " + s.iucn_override_source
        else:
            code, source = "NE", "missing"
        g.add((t, WIS.hasIUCNStatus, WIS[code]))
        g.add((t, WIS.statusSource, Literal(source)))

        for h in filter(None, s.habitats.split(";")):
            g.add((t, WIS.livesIn, WIS[h]))
        for th in filter(None, s.threats.split(";")):
            g.add((t, WIS.threatenedBy, WIS[th]))
        # Only 'Y' becomes a fact; '?' stays unknown (open-world) until GBIF data confirms it.
        if s.occurs_sg_seed == "Y":
            g.add((t, WIS.occursInCountry, WIS.SG))
        if s.occurs_penmy_seed == "Y":
            g.add((t, WIS.occursInCountry, WIS.MY))

    for _, z in zones.iterrows():
        zn = WIS[z.key]
        g.add((zn, RDF.type, WIS.ProtectedArea))
        g.add((zn, RDFS.label, Literal(z["name"], lang="en")))
        g.add((zn, WIS.inCountry, WIS[z.country]))
        g.add((zn, WIS.zoneType, Literal(z.zone_type)))
        g.add((zn, WIS.lat, Literal(z.lat, datatype=XSD.decimal)))
        g.add((zn, WIS.lon, Literal(z.lon, datatype=XSD.decimal)))
        for h in filter(None, z.habitats.split(";")):
            g.add((zn, WIS.hasHabitat, WIS[h]))
    return g


def main() -> None:
    g = build()
    g.serialize(SEED_TTL, format="turtle")
    print(f"wrote {SEED_TTL} ({len(g)} triples)")


if __name__ == "__main__":
    main()
