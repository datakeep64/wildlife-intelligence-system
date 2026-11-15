"""Shared recommender interface + ranking metrics.

Every architecture (knowledge-based, ALS collaborative filtering, content-based,
switching hybrid) implements fit/recommend, and is scored by the SAME
evaluate() on the SAME held-out split - otherwise "which one wins" is meaningless.

Interactions are a DataFrame with columns: user, item, weight (implicit feedback,
e.g. number of times an observer recorded a species).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class Rec:
    item: str
    score: float
    reasons: list[str] = field(default_factory=list)  # explanations (knowledge-based fills these)


class Recommender(ABC):
    name = "base"

    @abstractmethod
    def fit(self, interactions: pd.DataFrame) -> "Recommender":
        ...

    @abstractmethod
    def recommend(self, user: str | None = None, k: int = 10, context: dict | None = None,
                  exclude: set[str] | None = None) -> list[Rec]:
        """user=None means a cold-start visitor; `context` carries their stated constraints."""

    def evaluate(self, train: pd.DataFrame, test: pd.DataFrame, k: int = 10,
                 context_fn: Callable[[str], dict] | None = None, exclude_train: bool = True) -> dict:
        """precision@k, recall@k, hit-rate@k, catalogue coverage over users present in `test`."""
        seen = train.groupby("user")["item"].apply(set).to_dict()
        truth = test.groupby("user")["item"].apply(set).to_dict()
        p, r, hits, recommended = [], [], [], set()
        for u, rel in truth.items():
            excl = seen.get(u, set()) if exclude_train else set()
            recs = [x.item for x in self.recommend(user=u, k=k, context=context_fn(u) if context_fn else None, exclude=excl)]
            recommended.update(recs)
            n_hit = len(set(recs) & rel)
            p.append(n_hit / k)
            r.append(n_hit / len(rel))
            hits.append(n_hit > 0)
        catalogue = set(train["item"]) | set(test["item"])
        return {
            "model": self.name, "k": k, "n_users": len(truth),
            f"precision@{k}": round(float(np.mean(p)), 4) if p else None,
            f"recall@{k}": round(float(np.mean(r)), 4) if r else None,
            f"hit_rate@{k}": round(float(np.mean(hits)), 4) if hits else None,
            "coverage": round(len(recommended) / len(catalogue), 4) if catalogue else None,
        }


class PopularityRecommender(Recommender):
    """Baseline every other model must beat: most-observed items overall."""
    name = "popularity"

    def fit(self, interactions: pd.DataFrame) -> "PopularityRecommender":
        self.ranking = interactions.groupby("item")["user"].nunique().sort_values(ascending=False)
        return self

    def recommend(self, user=None, k=10, context=None, exclude=None) -> list[Rec]:
        exclude = exclude or set()
        items = [i for i in self.ranking.index if i not in exclude][:k]
        return [Rec(i, float(self.ranking[i]), ["popular with other observers"]) for i in items]


def leave_last_out_split(df: pd.DataFrame, time_col: str = "last_seen", min_items: int = 3,
                         test_frac: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user holdout: users with >= min_items items keep ~test_frac of their items
    (the most recent if time_col exists, else random) for testing."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for _, g in df.groupby("user"):
        if len(g) < min_items:
            train.append(g)
            continue
        g = g.sort_values(time_col) if time_col in g else g.sample(frac=1, random_state=int(rng.integers(1e9)))
        n_test = max(1, int(round(len(g) * test_frac)))
        train.append(g.iloc[:-n_test])
        test.append(g.iloc[-n_test:])
    return pd.concat(train), (pd.concat(test) if test else df.iloc[:0])
