"""Knowledge/constraint-based recommender on the ontology KG (cold start).

A new visitor has no history, so collaborative filtering has nothing to learn
from. Instead we reason over what we KNOW: which species occur in the visitor's
country, which habitats a zone has, which species live in those habitats, and
the visitor's stated preferences (e.g. "show me threatened species").

Constraints (hard filters) vs preferences (soft scores) is the classic
constraint-based recommender split. Every recommendation carries its reasons.

context = {
  "country": "SG" | "MY",          hard constraint
  "zone": "sungei_buloh",          optional; habitat match becomes a hard constraint
  "min_status": "VU",              optional hard constraint (IUCN floor)
  "prefer_threatened": True,       soft: boost by IUCN rank
  "habitats": ["mangrove"],        soft: boost species of these habitats
}
"""
import pandas as pd

from src.reasoning.kg.graph import load_graph, local, query
from src.reasoning.recsys.base import Rec, Recommender

STATUS_RANK = {"LC": 0, "NT": 1, "VU": 2, "EN": 3, "CR": 4}

SPECIES_FACTS = """
SELECT ?taxon ?name ?code ?rank ?habitat ?country WHERE {
  ?taxon a ?cls ; rdfs:label ?name ; wis:hasIUCNStatus ?st .
  VALUES ?cls { wis:Species wis:Subspecies }
  ?st wis:statusRank ?rank .
  BIND(STRAFTER(STR(?st), "#") AS ?code)
  OPTIONAL { ?taxon wis:livesIn ?habitat }
  OPTIONAL { ?taxon wis:occursInCountry ?country }
  FILTER(LANG(?name) = "en")
}
"""
ZONE_FACTS = """
SELECT ?zone ?name ?country ?habitat WHERE {
  ?zone a wis:ProtectedArea ; rdfs:label ?name ; wis:inCountry ?country ; wis:hasHabitat ?habitat .
}
"""


class KnowledgeBasedRecommender(Recommender):
    name = "knowledge_based"

    def __init__(self, w_habitat: float = 1.0, w_status: float = 0.5):
        self.w_habitat, self.w_status = w_habitat, w_status
        g = load_graph()
        sp = pd.DataFrame(query(SPECIES_FACTS, g))
        for c in ("taxon", "habitat", "country"):
            sp[c] = sp[c].map(lambda v: local(v) if v else None)
        self.species = {
            t: {"name": d.name.iloc[0], "code": d.code.iloc[0], "rank": int(d["rank"].iloc[0]),
                "habitats": set(d.habitat.dropna()), "countries": set(d.country.dropna())}
            for t, d in sp.groupby("taxon")
        }
        # A subspecies would duplicate its parent species (tiger vs Malayan tiger):
        # keep the more specific taxon only.
        parents = {local(o) for _, _, o in g.triples((None, g.namespace_manager.expand_curie("wis:subspeciesOf"), None))}
        for p in parents:
            self.species.pop(p, None)
        zn = pd.DataFrame(query(ZONE_FACTS, g))
        self.zones = {local(z): {"name": d.name.iloc[0], "country": local(d.country.iloc[0]),
                                 "habitats": {local(h) for h in d.habitat}} for z, d in zn.groupby("zone")}

    def fit(self, interactions=None) -> "KnowledgeBasedRecommender":
        return self  # nothing to learn - that is the point

    def recommend(self, user=None, k=10, context=None, exclude=None) -> list[Rec]:
        ctx, exclude = context or {}, exclude or set()
        zone = self.zones.get(ctx.get("zone")) if ctx.get("zone") else None
        country = ctx.get("country") or (zone["country"] if zone else None)
        min_rank = STATUS_RANK.get(ctx.get("min_status", "LC"), 0)
        pref_h = set(ctx.get("habitats", []))
        recs = []
        for key, s in self.species.items():
            if key in exclude:
                continue
            reasons = []
            # ---- hard constraints
            if country and country not in s["countries"]:
                continue
            if s["rank"] < min_rank:
                continue
            if zone:
                shared = s["habitats"] & zone["habitats"]
                if not shared:
                    continue
                reasons.append(f"lives in {', '.join(sorted(shared))}, found in {zone['name']}")
            elif country:
                reasons.append(f"recorded in {country}")
            # ---- soft preferences
            score = 0.0
            if zone:
                score += self.w_habitat * len(s["habitats"] & zone["habitats"]) / len(zone["habitats"])
            if pref_h & s["habitats"]:
                score += self.w_habitat * len(pref_h & s["habitats"]) / len(pref_h)
                reasons.append(f"matches your interest in {', '.join(sorted(pref_h & s['habitats']))}")
            if ctx.get("prefer_threatened") and s["rank"] >= 2:
                score += self.w_status * s["rank"] / 4
                reasons.append(f"IUCN {s['code']}")
            recs.append(Rec(key, round(score, 4), reasons))
        recs.sort(key=lambda r: (-r.score, r.item))
        return recs[:k]

    def recommend_zones(self, context: dict | None = None, k: int = 3, per_zone: int = 10) -> list[Rec]:
        """Trail suggestions: rank zones by the total score of the species they could hold."""
        ctx = context or {}
        out = []
        for zkey, z in self.zones.items():
            if ctx.get("country") and z["country"] != ctx["country"]:
                continue
            recs = self.recommend(k=per_zone, context={**ctx, "zone": zkey})
            score = sum(r.score for r in recs)
            out.append(Rec(zkey, round(score, 4), [f"{len(recs)} matching species, e.g. " +
                                                     ", ".join(self.species[r.item]["name"] for r in recs[:3])]))
        out.sort(key=lambda r: -r.score)
        return out[:k]


if __name__ == "__main__":
    kb = KnowledgeBasedRecommender()
    ctx = {"country": "SG", "zone": "sungei_buloh", "prefer_threatened": True}
    print(f"Visitor context: {ctx}")
    for r in kb.recommend(k=5, context=ctx):
        print(f"  {kb.species[r.item]['name']:28s} {r.score:.2f}  <- {'; '.join(r.reasons)}")
    print("\nTrail suggestions (SG, threatened-first):")
    for z in kb.recommend_zones({"country": "SG", "prefer_threatened": True}):
        print(f"  {kb.zones[z.item]['name']:36s} {z.score:.2f}  {z.reasons[0]}")
