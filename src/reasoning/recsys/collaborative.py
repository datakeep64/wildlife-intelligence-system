"""Collaborative filtering: implicit ALS on the observer x species matrix.

Learns latent 'taste' vectors from who recorded what. Strength: active observers
with long histories. Weakness: a user with no history has no vector, so it cannot
recommend anything to them (cold start); that failure is measured, not patched here.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.reasoning.recsys.base import Rec, Recommender


class ALSRecommender(Recommender):
    name = "collaborative_als"

    def __init__(self, factors: int = 64, regularization: float = 0.05, alpha: float = 10.0,
                 iterations: int = 20, seed: int = 42):
        self.params = dict(factors=factors, regularization=regularization, alpha=alpha, iterations=iterations)
        self.seed = seed

    def fit(self, interactions: pd.DataFrame) -> "ALSRecommender":
        from implicit.als import AlternatingLeastSquares

        self.users = {u: i for i, u in enumerate(interactions.user.unique())}
        self.items = np.array(sorted(interactions.item.unique()))
        item_idx = {it: i for i, it in enumerate(self.items)}
        rows = interactions.user.map(self.users).to_numpy()
        cols = interactions.item.map(item_idx).to_numpy()
        vals = np.log1p(interactions.weight.to_numpy()).astype(np.float32)  # dampen prolific observers
        self.ui = csr_matrix((vals, (rows, cols)), shape=(len(self.users), len(self.items)))
        self.model = AlternatingLeastSquares(**self.params, random_state=self.seed)
        self.model.fit(self.ui, show_progress=False)
        return self

    def recommend(self, user=None, k=10, context=None, exclude=None) -> list[Rec]:
        if user not in self.users:
            return []  # cold start: no latent vector
        uid = self.users[user]
        ids, scores = self.model.recommend(uid, self.ui[uid], N=k + len(exclude or ()), filter_already_liked_items=True)
        out = [Rec(self.items[i], float(s), ["observers with similar histories recorded it"])
               for i, s in zip(ids, scores) if self.items[i] not in (exclude or set())]
        return out[:k]
