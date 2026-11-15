"""Unit tests for the recommender split logic and switching hybrid (toy data, no downloads)."""
import pandas as pd

from src.reasoning.recsys.base import PopularityRecommender, Rec, Recommender
from src.reasoning.recsys.data import make_split, segment
from src.reasoning.recsys.hybrid import SwitchingHybrid


def toy_records():
    rows = [
        ("a", "x", "2025-03-01"), ("a", "y", "2025-04-01"), ("a", "z", "2026-02-01"), ("a", "x", "2026-02-02"),
        ("b", "x", "2026-03-01"),                       # new user after the cutoff -> cold
        ("c", "y", "2025-05-01"), ("c", "y", "2026-01-05"),  # only re-sees a known species -> no target
    ]
    d = pd.DataFrame(rows, columns=["user", "item", "date"])
    d["date"] = pd.to_datetime(d.date)
    d["region"], d["gbifID"] = "SG", range(len(d))
    return d


def test_split_targets_are_new_species_only():
    s = make_split(toy_records(), "2026-01-01")
    assert s.truth == {"a": {"z"}, "b": {"x"}}      # 'x' for a was already seen; c has nothing new
    assert s.segment == {"a": "light", "b": "cold"}
    assert s.context["b"] == {"country": "SG"}


def test_segments():
    assert [segment(n) for n in (0, 1, 4, 5, 50)] == ["cold", "light", "light", "heavy", "heavy"]


class Fixed(Recommender):
    def __init__(self, item):
        self.item = item

    def fit(self, interactions):
        return self

    def recommend(self, user=None, k=10, context=None, exclude=None):
        return [Rec(self.item, 1.0)]


def test_switching_hybrid_routes_by_segment():
    h = SwitchingHybrid({"m1": Fixed("x"), "m2": Fixed("y")}, {"cold": "m1", "light": "m2", "heavy": "m2"},
                        {"u_light": "light"})
    assert h.recommend("u_light")[0].item == "y"
    assert h.recommend("unknown_user")[0].item == "x"   # unseen users are treated as cold


def test_popularity_excludes_seen():
    inter = pd.DataFrame({"user": ["a", "b", "b"], "item": ["x", "x", "y"], "weight": 1})
    recs = PopularityRecommender().fit(inter).recommend(k=2, exclude={"x"})
    assert [r.item for r in recs] == ["y"]
