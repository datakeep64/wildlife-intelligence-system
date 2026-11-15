"""Property graph of occurrence data (NetworkX) and its export to KG triples.

Nodes : species, grid squares (0.1 deg, about 11 km), families, orders, classes, regions,
        IUCN statuses, habitats.
Edges : species -recordedIn-> square   (attributes: records, observers, first_year)
        square  -adjacentTo-> square   (8-neighbourhood, both directions)
        square  -inRegion-> SG | PenMY
        species -inFamily-> family -inOrder-> order -inClass-> class
        species -hasStatus-> IUCN category        (from GBIF)
        species -hasHabitat-> habitat              (from the curated ontology KG, 25 species)

Why a property graph here and not the RDF ontology: occurrence data is large and messy, and the
questions asked of it are graph analytics (co-occurrence, survey effort, neighbourhoods), not
logical entailment. The curated KG contributes what it is good at: habitats and status.
"""
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

from src.common.paths import GBIF_DIR, SPECIES_CSV

RES = 0.1                                  # grid resolution in degrees (about 11 km)
MAX_UNC_M = RES * 111_000 / 2              # precision guard: drop records blurred beyond half a square
THREATENED = {"VU", "EN", "CR"}


def cell_of(lat, lon) -> pd.Series:
    iy = np.floor(np.asarray(lat, dtype=float) / RES).astype(int)
    ix = np.floor(np.asarray(lon, dtype=float) / RES).astype(int)
    return pd.Series([f"sq_{a}_{b}" for a, b in zip(iy, ix)])


def cell_centre(cell: str) -> tuple[float, float]:
    _, iy, ix = cell.split("_")
    return (int(iy) + 0.5) * RES, (int(ix) + 0.5) * RES


def neighbours(cell: str) -> list[str]:
    _, iy, ix = cell.split("_")
    iy, ix = int(iy), int(ix)
    return [f"sq_{iy + dy}_{ix + dx}" for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]


def load_occurrences() -> pd.DataFrame:
    """Observer records + the curated species' own download, deduplicated, with grid squares."""
    frames = [pd.read_csv(GBIF_DIR / f, low_memory=False)
              for f in ("observers_occurrences.csv", "seed_species_occurrences.csv")]
    d = pd.concat(frames, ignore_index=True).drop_duplicates("gbifID")
    d = d.dropna(subset=["species", "decimalLatitude", "decimalLongitude", "year"])
    n_all = len(d)
    # missing uncertainty is treated as precise, as in reconcile.py
    d = d[d.coordinateUncertaintyInMeters.fillna(0) <= MAX_UNC_M].copy()
    d.attrs["dropped_imprecise"] = n_all - len(d)
    d["year"] = d.year.astype(int)
    d["cell"] = cell_of(d.decimalLatitude, d.decimalLongitude).values
    return d


def species_info(d: pd.DataFrame) -> pd.DataFrame:
    """One row per species: taxonomy, most common IUCN category, curated habitats."""
    mode = lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None  # noqa: E731
    info = d.groupby("species").agg(family=("family", mode), order=("order", mode),
                                    cls=("class", mode), iucn=("iucnRedListCategory", mode))
    cur = pd.read_csv(SPECIES_CSV)
    cur = cur[cur["rank"] == "species"].set_index("scientific_name")
    info["habitats"] = [cur.habitats.get(s, None) if s in cur.index else None for s in info.index]
    return info


def pairs(d: pd.DataFrame) -> pd.DataFrame:
    """Species-square links with their evidence."""
    return (d.groupby(["species", "cell"])
            .agg(records=("gbifID", "size"), observers=("recordedBy", "nunique"),
                 first_year=("year", "min"), region=("region", "first"))
            .reset_index())


# ------------------------------------------------------------------ property graph
def build_graph(links: pd.DataFrame, info: pd.DataFrame) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    cells = set(links.cell)
    for s in links.species.unique():
        r = info.loc[s]
        g.add_node(s, kind="species", iucn=r.iucn, family=r.family)
    for c, reg in links.groupby("cell").region.first().items():
        lat, lon = cell_centre(c)
        g.add_node(c, kind="square", lat=lat, lon=lon, region=reg)
    for row in links.itertuples(index=False):
        g.add_edge(row.species, row.cell, key="recordedIn", records=row.records,
                   observers=row.observers, first_year=row.first_year)
    for c in cells:
        for n in neighbours(c):
            if n in cells:
                g.add_edge(c, n, key="adjacentTo")
    return g


def cooccurrence(links: pd.DataFrame, min_shared: int = 3) -> pd.DataFrame:
    """Species pairs sharing squares (bipartite projection), with Jaccard similarity."""
    sets = links.groupby("species").cell.apply(frozenset)
    by_cell = links.groupby("cell").species.apply(list)
    shared = Counter()
    for sp in by_cell:
        sp = sorted(sp)
        for i in range(len(sp)):
            for j in range(i + 1, len(sp)):
                shared[(sp[i], sp[j])] += 1
    rows = [(a, b, n, n / len(sets[a] | sets[b])) for (a, b), n in shared.items() if n >= min_shared]
    return pd.DataFrame(rows, columns=["a", "b", "shared", "jaccard"]).sort_values("jaccard", ascending=False)


def effort_summary(d: pd.DataFrame, links: pd.DataFrame) -> dict:
    """How uneven is survey effort across squares?"""
    rec = d.groupby("cell").size().sort_values(ascending=False)
    sp = links.groupby("cell").species.nunique()
    top10 = rec.iloc[: max(1, len(rec) // 10)].sum() / rec.sum()
    return {
        "squares": int(len(rec)),
        "records_share_top10pct_squares": round(float(top10), 3),
        "median_species_per_square": float(sp.median()),
        "squares_with_1_species": int((sp == 1).sum()),
        "corr_records_vs_species": round(float(np.corrcoef(np.log1p(rec[sp.index]), sp)[0, 1]), 3),
    }


# ------------------------------------------------------------------ KG triples for PyKEEN
RELATION_GROUPS = {
    "occurrence": ["recordedIn"],
    "space": ["adjacentTo", "inRegion"],
    "taxonomy": ["inFamily", "inOrder", "inClass"],
    "species_facts": ["hasStatus", "hasHabitat"],
}


def triples(links: pd.DataFrame, info: pd.DataFrame, groups=tuple(RELATION_GROUPS)) -> np.ndarray:
    """Labelled (head, relation, tail) triples from the training links only."""
    t = []
    species = links.species.unique()
    cells = set(links.cell)
    if "occurrence" in groups:
        t += [(s, "recordedIn", c) for s, c in links[["species", "cell"]].itertuples(index=False)]
    if "space" in groups:
        t += [(c, "adjacentTo", n) for c in cells for n in neighbours(c) if n in cells]
        t += [(c, "inRegion", f"region_{r}") for c, r in links.groupby("cell").region.first().items()]
    if "taxonomy" in groups:
        sub = info.loc[species]
        t += [(s, "inFamily", f"fam_{r.family}") for s, r in sub.iterrows() if r.family]
        fam = sub.dropna(subset=["family", "order"]).drop_duplicates("family")
        t += [(f"fam_{r.family}", "inOrder", f"ord_{r.order}") for _, r in fam.iterrows()]
        ords = sub.dropna(subset=["order", "cls"]).drop_duplicates("order")
        t += [(f"ord_{r.order}", "inClass", f"cls_{r.cls}") for _, r in ords.iterrows()]
    if "species_facts" in groups:
        sub = info.loc[species]
        t += [(s, "hasStatus", f"iucn_{r.iucn}") for s, r in sub.iterrows() if r.iucn and r.iucn != "NE"]
        for s, r in sub.iterrows():
            if isinstance(r.habitats, str):
                t += [(s, "hasHabitat", f"hab_{h}") for h in r.habitats.split(";")]
    return np.array(t, dtype=str)
