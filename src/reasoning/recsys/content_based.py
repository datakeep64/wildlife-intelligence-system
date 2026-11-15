"""Content-based: recommend species similar to the ones a user has already recorded.

Two feature sets for all species in the catalogue:
  taxonomy : one-hot class / order / family (weighted 0.25 / 0.5 / 1.0) + range
             (share of records in SG, mean lat/lon, from TRAINING records only)
  bioclip  : BioCLIP text embedding of the species' scientific name ("a photo of <name>"),
             the same model as the camera pipeline. Tests whether a vision-language
             representation knows which species belong together.
User profile = log-weighted mean of their species' vectors; score = cosine similarity.
"""
import numpy as np
import pandas as pd

from src.reasoning.recsys.base import Rec, Recommender


def taxonomy_features(records: pd.DataFrame, train_records: pd.DataFrame) -> pd.DataFrame:
    tax = records.groupby("item")[["class", "order", "family"]].first()
    parts = [pd.get_dummies(tax[c], prefix=c).astype(float) * w for c, w in (("class", 0.25), ("order", 0.5), ("family", 1.0))]
    rng = train_records.groupby("item").agg(sg=("region", lambda r: (r == "SG").mean()),
                                            lat=("decimalLatitude", "mean"), lon=("decimalLongitude", "mean"))
    rng = rng.reindex(tax.index)
    rng = rng.fillna(rng.mean())
    rng = (rng - rng.mean()) / rng.std().replace(0, 1) * 0.5
    return pd.concat(parts + [rng], axis=1)


def bioclip_features(items: list[str], model_name: str = "hf-hub:imageomics/bioclip") -> pd.DataFrame:
    from src.perception.camera_trap.embed import BioCLIP

    clip = BioCLIP(model_name)
    emb = np.concatenate([clip.text_embeddings([f"a photo of {s}" for s in items[i:i + 128]]) for i in range(0, len(items), 128)])
    return pd.DataFrame(emb, index=items)


class ContentBasedRecommender(Recommender):
    def __init__(self, features: pd.DataFrame, name: str):
        X = features.to_numpy(dtype=np.float32, copy=True)
        X /= np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-9)
        self.X, self.items, self.name = X, np.array(features.index), name
        self.idx = {it: i for i, it in enumerate(self.items)}

    def fit(self, interactions: pd.DataFrame) -> "ContentBasedRecommender":
        self.profiles = {}
        for u, g in interactions.groupby("user"):
            ids = [self.idx[i] for i in g.item if i in self.idx]
            if ids:
                w = np.log1p(g.weight[g.item.isin(self.idx)].to_numpy())[:, None]
                p = (self.X[ids] * w).sum(0)
                self.profiles[u] = p / max(np.linalg.norm(p), 1e-9)
        return self

    def recommend(self, user=None, k=10, context=None, exclude=None) -> list[Rec]:
        if user not in self.profiles:
            return []  # no history -> nothing to be similar to
        scores = self.X @ self.profiles[user]
        exclude = exclude or set()
        out = []
        for i in np.argsort(-scores):
            if self.items[i] not in exclude:
                out.append(Rec(self.items[i], float(scores[i]), ["similar to species you have recorded"]))
                if len(out) == k:
                    break
        return out
