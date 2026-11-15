"""Predict where species will be recorded next: species -recordedIn-> square link prediction.

Time split. Train on every link first recorded up to `last_train_year`; test on links first
recorded in the following years. A test link can only be scored if its species and its square
are already in the training graph (embeddings are transductive); the rest are counted, not hidden.

Ranking is "filtered": for a test link (s, c), c is ranked against every square where s has
NOT been recorded in training and that is not another test answer for s. Ties count half.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .property_graph import neighbours, pairs


@dataclass
class LPSplit:
    train: pd.DataFrame                 # species-square links known by the end of last_train_year
    test: pd.DataFrame                  # new links first recorded in the test years (scorable)
    species: list[str]
    cells: list[str]
    excluded: dict = field(default_factory=dict)
    label: str = ""

    def __post_init__(self):
        self.s_idx = {s: i for i, s in enumerate(self.species)}
        self.c_idx = {c: i for i, c in enumerate(self.cells)}

    def incidence(self) -> np.ndarray:
        X = np.zeros((len(self.species), len(self.cells)), dtype=np.float32)
        X[[self.s_idx[s] for s in self.train.species], [self.c_idx[c] for c in self.train.cell]] = 1
        return X


def make_split(d: pd.DataFrame, last_train_year: int, test_years: tuple[int, ...]) -> LPSplit:
    train = pairs(d[d.year <= last_train_year])
    first = pairs(d)
    new = first[first.first_year.isin(test_years)]
    s_ok, c_ok = new.species.isin(set(train.species)), new.cell.isin(set(train.cell))
    excluded = {"new_links": int(len(new)), "unseen_species": int((~s_ok).sum()),
                "unseen_square": int((s_ok & ~c_ok).sum())}
    test = new[s_ok & c_ok].reset_index(drop=True)
    return LPSplit(train, test, sorted(train.species.unique()), sorted(train.cell.unique()), excluded,
                   f"train<={last_train_year}, test {test_years[0]}-{test_years[-1]}")


# ------------------------------------------------------------------ evaluation
def rank_test(split: LPSplit, scores: np.ndarray) -> pd.DataFrame:
    """scores: [n_species, n_cells] from any method. Returns one row per test link with its rank."""
    X = split.incidence().astype(bool)
    tests = split.test.groupby("species").cell.apply(list)
    rows = []
    for s, cells in tests.items():
        i = split.s_idx[s]
        t_idx = np.array([split.c_idx[c] for c in cells])
        cand = ~X[i]
        cand[t_idx] = False                       # other answers for s are filtered out
        pool = scores[i, cand]
        for c, j in zip(cells, t_idx):
            v = scores[i, j]
            rank = 1 + (pool > v).sum() + 0.5 * (pool == v).sum()
            rows.append((s, c, float(rank), int(cand.sum()) + 1))
    return pd.DataFrame(rows, columns=["species", "cell", "rank", "n_candidates"])


def metrics(ranks: pd.DataFrame) -> dict:
    r = ranks["rank"].to_numpy()
    return {"mrr": float(np.mean(1 / r)), "hits@10": float(np.mean(r <= 10)),
            "hits@50": float(np.mean(r <= 50)), "median_rank": float(np.median(r)), "n": int(len(r))}


# ------------------------------------------------------------------ baselines
def _tiebreak(X: np.ndarray) -> np.ndarray:
    pop = X.sum(0)
    return pop / (pop.max() + 1)                   # in [0, 1): only ever breaks ties


def popularity_scores(split: LPSplit) -> np.ndarray:
    """Rank squares by how many species they already hold, the same for every species.
    This mostly measures survey effort."""
    X = split.incidence()
    return np.tile(X.sum(0), (X.shape[0], 1))


def cooccurrence_scores(split: LPSplit, k: int = 25) -> np.ndarray:
    """Item-kNN on the property graph: a square scores highly if the species' closest
    co-occurring species (Jaccard over squares) are recorded there."""
    X = split.incidence()
    inter = X @ X.T
    size = X.sum(1)
    sim = inter / np.maximum(size[:, None] + size[None, :] - inter, 1)
    np.fill_diagonal(sim, 0)
    if k < sim.shape[1]:
        kth = -np.partition(-sim, k - 1, axis=1)[:, k - 1:k]
        sim = np.where(sim >= kth, sim, 0)
    return sim @ X + 1e-3 * _tiebreak(X)


def spatial_scores(split: LPSplit) -> np.ndarray:
    """Range expansion: a square scores by how many of its 8 neighbours already hold the species."""
    X = split.incidence()
    A = np.zeros((len(split.cells), len(split.cells)), dtype=np.float32)
    for c, j in split.c_idx.items():
        for n in neighbours(c):
            if n in split.c_idx:
                A[split.c_idx[n], j] = 1
    return X @ A + 1e-3 * _tiebreak(X)
