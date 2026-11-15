"""Property graph + link prediction: split hygiene, filtered ranking, baselines (toy data, no downloads)."""
import numpy as np
import pandas as pd
import pytest

from src.reasoning.graph.link_prediction import (LPSplit, cooccurrence_scores, make_split, metrics, popularity_scores,
                                                 rank_test, spatial_scores)
from src.reasoning.graph.property_graph import cell_of, neighbours, triples


def rec(species, cell, year, n=1):
    return [{"gbifID": f"{species}{cell}{year}{i}", "species": species, "cell": cell, "year": year,
             "recordedBy": "o", "region": "SG"} for i in range(n)]


@pytest.fixture
def toy():
    rows = (rec("a", "sq_0_0", 2020) + rec("a", "sq_0_1", 2021) + rec("b", "sq_0_0", 2020) + rec("b", "sq_0_1", 2020)
            + rec("b", "sq_5_5", 2022) + rec("c", "sq_5_5", 2021)
            + rec("a", "sq_5_5", 2025)            # new link, both entities known -> test
            + rec("a", "sq_0_0", 2025)            # already known -> not a test link
            + rec("d", "sq_0_0", 2025)            # unseen species -> excluded, counted
            + rec("c", "sq_9_9", 2026))           # unseen square -> excluded, counted
    return pd.DataFrame(rows)


def test_cells_and_neighbours():
    assert list(cell_of([1.35, -0.05], [103.82, 103.85])) == ["sq_13_1038", "sq_-1_1038"]
    assert len(neighbours("sq_0_0")) == 8 and "sq_0_0" not in neighbours("sq_0_0")


def test_time_split_has_no_leakage(toy):
    sp = make_split(toy, 2024, (2025, 2026))
    assert list(zip(sp.test.species, sp.test.cell)) == [("a", "sq_5_5")]
    assert sp.excluded == {"new_links": 3, "unseen_species": 1, "unseen_square": 1}
    assert not set(zip(sp.test.species, sp.test.cell)) & set(zip(sp.train.species, sp.train.cell))
    info = pd.DataFrame({"family": None, "order": None, "cls": None, "iucn": None, "habitats": None},
                        index=sp.species)
    tri = {tuple(t) for t in triples(sp.train, info)}
    assert ("a", "recordedIn", "sq_5_5") not in tri   # the answer is never in the training graph


def test_filtered_rank_ignores_known_squares_and_counts_ties_half(toy):
    sp = make_split(toy, 2024, (2025, 2026))
    i, j = sp.s_idx["a"], sp.c_idx["sq_5_5"]
    s = np.zeros((len(sp.species), len(sp.cells)))
    s[i, :] = 9.0                       # known squares score higher, but are filtered out
    s[i, j] = 1.0
    r = rank_test(sp, s)
    assert r["rank"].tolist() == [1.0] and metrics(r)["mrr"] == 1.0

    # ties: answer tied with 2 candidates and beaten by 1 -> rank 1 + 1 + 0.5 * 2 = 3
    tr = pd.DataFrame({"species": ["a"], "cell": ["c0"]})
    sp2 = LPSplit(tr, pd.DataFrame({"species": ["a"], "cell": ["c1"]}), ["a"], ["c0", "c1", "c2", "c3", "c4"])
    assert rank_test(sp2, np.array([[99, 5, 5, 5, 7]]))["rank"].tolist() == [3.0]


def test_baselines_rank_the_obvious_square_first():
    # species x is in sq_0_0; its neighbour sq_0_1 should beat a far square for the spatial baseline;
    # y always co-occurs with x and is also in sq_7_7, so co-occurrence should point x to sq_7_7.
    rows = (rec("x", "sq_0_0", 2020) + rec("y", "sq_0_0", 2020) + rec("y", "sq_7_7", 2020)
            + rec("z", "sq_0_1", 2020) + rec("z", "sq_3_3", 2020) + rec("w", "sq_3_3", 2020)
            + rec("x", "sq_7_7", 2025) + rec("x", "sq_0_1", 2025))
    sp = make_split(pd.DataFrame(rows), 2024, (2025,))
    x = sp.s_idx["x"]
    co, spa = cooccurrence_scores(sp, k=1)[x], spatial_scores(sp)[x]
    assert co[sp.c_idx["sq_7_7"]] > co[sp.c_idx["sq_3_3"]]
    assert spa[sp.c_idx["sq_0_1"]] > spa[sp.c_idx["sq_7_7"]]
    pop = popularity_scores(sp)
    assert (pop == pop[0]).all()       # popularity is the same list for every species
