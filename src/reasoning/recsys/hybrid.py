"""Regional popularity, knowledge-based adapters, and the switching hybrid."""
from functools import lru_cache

import pandas as pd

from src.common.paths import SPECIES_CSV
from src.reasoning.recsys.base import Rec, Recommender
from src.reasoning.recsys.data import REGION_TO_COUNTRY
from src.reasoning.recsys.knowledge_based import KnowledgeBasedRecommender


class RegionalPopularity(Recommender):
    """Most-observed species in the user's country. Uses context only, so it also serves cold users."""
    name = "popularity_regional"

    def __init__(self, records: pd.DataFrame):
        self.records = records

    def fit(self, interactions: pd.DataFrame) -> "RegionalPopularity":
        users = set(interactions.user)
        r = self.records[self.records.user.isin(users)]
        self.rank = {REGION_TO_COUNTRY[reg]: g.groupby("item")["user"].nunique().sort_values(ascending=False)
                     for reg, g in r.groupby("region")}
        return self

    def recommend(self, user=None, k=10, context=None, exclude=None) -> list[Rec]:
        ranking = self.rank[(context or {}).get("country", "SG")]
        exclude = exclude or set()
        return [Rec(i, float(n), [f"popular with observers in {(context or {}).get('country', 'SG')}"])
                for i, n in ranking.items() if i not in exclude][:k]


def kg_to_catalogue() -> dict[str, str]:
    """KG key -> GBIF species name. A subspecies maps to its parent species (GBIF records are species-level)."""
    sp = pd.read_csv(SPECIES_CSV, dtype=str).fillna("").set_index("key")
    return {k: (sp.loc[r.parent, "scientific_name"] if r.parent else r.scientific_name) for k, r in sp.iterrows()}


class KnowledgeAdapter(Recommender):
    """Wraps the ontology recommender for the GBIF catalogue.

    variant="pure":       KG decides eligibility AND order (habitat breadth + threatened preference)
    variant="popularity": KG decides eligibility and explanation; regional popularity decides order
    """
    def __init__(self, variant: str, regional: RegionalPopularity | None = None):
        assert variant in ("pure", "popularity")
        self.variant, self.regional = variant, regional
        self.name = f"knowledge_{variant}"
        self.kb = KnowledgeBasedRecommender()
        self.to_cat = kg_to_catalogue()

    def fit(self, interactions: pd.DataFrame) -> "KnowledgeAdapter":
        self._eligible.cache_clear()
        return self

    @lru_cache(maxsize=4)
    def _eligible(self, country: str) -> tuple:
        ctx = {"country": country}
        if self.variant == "pure":
            ctx |= {"prefer_generalists": True, "prefer_threatened": True}
        recs = self.kb.recommend(k=100, context=ctx)
        seen, out = set(), []
        for r in recs:
            item = self.to_cat[r.item]
            if item not in seen:
                seen.add(item)
                out.append(Rec(item, r.score, r.reasons))
        if self.variant == "popularity":
            rank = self.regional.rank.get(country, pd.Series(dtype=float))
            out.sort(key=lambda r: -rank.get(r.item, 0))
        return tuple(out)

    def recommend(self, user=None, k=10, context=None, exclude=None) -> list[Rec]:
        exclude = exclude or set()
        return [r for r in self._eligible((context or {}).get("country", "SG")) if r.item not in exclude][:k]


class SwitchingHybrid(Recommender):
    """Routes each user to the model chosen for their segment (cold / light / heavy).
    The policy is chosen on a VALIDATION split, never on the test set."""
    name = "switching_hybrid"

    def __init__(self, models: dict[str, Recommender], policy: dict[str, str], segment_of: dict[str, str]):
        self.models, self.policy, self.segment_of = models, policy, segment_of

    def fit(self, interactions: pd.DataFrame) -> "SwitchingHybrid":
        return self  # component models are fitted already

    def recommend(self, user=None, k=10, context=None, exclude=None) -> list[Rec]:
        seg = self.segment_of.get(user, "cold")
        return self.models[self.policy[seg]].recommend(user, k, context, exclude)
