"""Observer x species interactions from iNaturalist (via GBIF), with a time-based split.

Item  = species (scientific name).  User = observer (`recordedBy`).
Weight = number of records of that species by that observer (implicit feedback).

Time-based split: train on records before the cutoff, test on records after it.
The test target for a user is the set of species they recorded after the cutoff that
were NOT already in their training history ("what will they see next that's new to them").
Users with no training history are the natural cold-start users.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.paths import GBIF_DIR

REGION_TO_COUNTRY = {"SG": "SG", "PenMY": "MY"}


def load_records() -> pd.DataFrame:
    d = pd.read_csv(GBIF_DIR / "observers_occurrences.csv", low_memory=False)
    d = d.dropna(subset=["species", "recordedBy", "eventDate"])
    d["date"] = pd.to_datetime(d.eventDate.str[:10], errors="coerce")
    d = d.dropna(subset=["date"])
    return d.rename(columns={"recordedBy": "user", "species": "item"})


def interactions(records: pd.DataFrame) -> pd.DataFrame:
    return (records.groupby(["user", "item"])
            .agg(weight=("gbifID", "size"), last_seen=("date", "max"))
            .reset_index())


def segment(n_train_items: int) -> str:
    if n_train_items == 0:
        return "cold"
    return "light" if n_train_items < 5 else "heavy"


@dataclass
class Split:
    train: pd.DataFrame            # interactions before cutoff
    truth: dict[str, set]          # user -> NEW species recorded after cutoff
    seen: dict[str, set]           # user -> species in training history
    context: dict[str, dict]       # user -> {"country": ...} (where their first post-cutoff record was)
    segment: dict[str, str]        # user -> cold / light / heavy
    cutoff: str
    end: str | None


def make_split(records: pd.DataFrame, cutoff: str, end: str | None = None) -> Split:
    before = records[records.date < cutoff]
    after = records[records.date >= cutoff]
    if end:
        after = after[after.date < end]
    train = interactions(before)
    seen = train.groupby("user")["item"].apply(set).to_dict()
    truth, context, seg = {}, {}, {}
    for u, g in after.sort_values("date").groupby("user"):
        new = set(g["item"]) - seen.get(u, set())
        if not new:
            continue
        truth[u] = new
        context[u] = {"country": REGION_TO_COUNTRY[g.region.iloc[0]]}
        seg[u] = segment(len(seen.get(u, ())))
    return Split(train, truth, seen, context, seg, cutoff, end)


def item_popularity(train: pd.DataFrame) -> pd.Series:
    """Share of training users who recorded each item."""
    return train.groupby("item")["user"].nunique() / train.user.nunique()


def novelty(items: list[str], pop: pd.Series) -> float:
    """Mean self-information of recommended items: higher = less obvious recommendations."""
    floor = 1 / (len(pop) * 10)
    return float(np.mean([-np.log2(pop.get(i, floor)) for i in items])) if items else float("nan")
