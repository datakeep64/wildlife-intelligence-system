"""Fast tests for the ontology KG and recommender framework (no downloads needed
beyond species_seed.ttl, which is committed)."""
import pandas as pd

from src.reasoning.kg.graph import THREATENED_TAXA, query
from src.reasoning.recsys.base import PopularityRecommender, leave_last_out_split
from src.reasoning.recsys.knowledge_based import KnowledgeBasedRecommender


def test_malayan_tiger_is_cr_via_override():
    rows = {r["name"]: r["code"] for r in query(THREATENED_TAXA)}
    assert rows["Malayan tiger"] == "CR"  # subspecies override
    assert rows["Tiger"] == "EN"          # species-level status from GBIF


def test_multilingual_labels_present():
    r = query('SELECT ?l WHERE { wis:malayan_tapir rdfs:label ?l FILTER(LANG(?l)="ms") }')
    assert r and r[0]["l"] == "tenuk"


def test_knowledge_based_respects_hard_constraints():
    kb = KnowledgeBasedRecommender()
    recs = kb.recommend(k=50, context={"country": "SG", "zone": "sungei_buloh", "min_status": "VU"})
    assert recs, "expected at least one SG wetland/mangrove species"
    for r in recs:
        s = kb.species[r.item]
        assert "SG" in s["countries"] and s["rank"] >= 2
        assert s["habitats"] & kb.zones["sungei_buloh"]["habitats"]
        assert r.reasons


def test_subspecies_replaces_parent():
    kb = KnowledgeBasedRecommender()
    assert "malayan_tiger" in kb.species and "tiger" not in kb.species


def test_metrics_on_toy_data():
    df = pd.DataFrame({"user": list("aaaabbbbcc"), "item": list("wxyzwxyqwx"), "weight": 1})
    train, test = leave_last_out_split(df, min_items=3, test_frac=0.25)
    m = PopularityRecommender().fit(train).evaluate(train, test, k=2)
    assert m["n_users"] == 2 and 0 <= m["precision@2"] <= 1 and 0 <= m["recall@2"] <= 1
