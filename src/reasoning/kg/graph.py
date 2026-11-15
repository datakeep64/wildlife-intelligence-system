"""Load the ontology KG (TBox + ABox) and run SPARQL queries."""
from functools import lru_cache

from rdflib import Graph, Namespace

from src.common.paths import ONTOLOGY_TTL, SEED_TTL

WIS = Namespace("https://datakeep64.github.io/wis/ontology#")
PREFIXES = """
PREFIX wis:  <https://datakeep64.github.io/wis/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
"""


OBS_TTL = SEED_TTL.with_name("observations.ttl")


@lru_cache(maxsize=2)
def load_graph(include_observations: bool = True) -> Graph:
    """TBox + seed ABox (+ GBIF-reconciled observations). Cached: treat as read-only."""
    g = Graph()
    g.parse(ONTOLOGY_TTL, format="turtle")
    if not SEED_TTL.exists():
        raise FileNotFoundError(f"{SEED_TTL} missing - run: python -m src.reasoning.kg.build_seed")
    g.parse(SEED_TTL, format="turtle")
    if include_observations and OBS_TTL.exists():
        g.parse(OBS_TTL, format="turtle")
    g.bind("wis", WIS)
    return g


def working_copy(include_observations: bool = True) -> Graph:
    """A mutable copy for rules to write into (never mutate the cached graph)."""
    g = Graph()
    g += load_graph(include_observations)
    g.bind("wis", WIS)
    return g


def query(sparql: str, g: Graph | None = None) -> list[dict]:
    """Run a SPARQL SELECT and return rows as dicts of Python values."""
    g = g or load_graph()
    res = g.query(PREFIXES + sparql)
    return [{str(v): (row[v].toPython() if row[v] is not None else None) for v in res.vars} for row in res]


def local(uri) -> str:
    """wis:malayan_tiger -> 'malayan_tiger'."""
    return str(uri).split("#")[-1]


THREATENED_TAXA = """
SELECT ?taxon ?name ?code ?rank WHERE {
  ?taxon wis:hasIUCNStatus ?status ; rdfs:label ?name .
  ?status wis:statusRank ?rank .
  BIND(STRAFTER(STR(?status), "#") AS ?code)
  FILTER(?rank >= 2 && LANG(?name) = "en")
} ORDER BY DESC(?rank) ?name
"""

if __name__ == "__main__":
    for r in query(THREATENED_TAXA):
        print(f"{r['code']}  {r['name']}")
