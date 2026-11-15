"""Dataset config + leakage-safe train/test split for the camera-trap experiment."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def location_split(df: pd.DataFrame, test_fraction: float = 0.25, seed: int = 42) -> pd.Series:
    """Assign 'train'/'test' so that no camera location appears in both.

    Why: images from the same camera share background, angle and lighting.
    A random split lets a model 'recognise the background' and inflates accuracy.
    StratifiedGroupKFold keeps class balance roughly even while grouping by location.
    """
    n_splits = max(2, round(1 / test_fraction))
    groups = df["location"].fillna("unknown_" + df["image_id"].astype(str))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, test_idx = next(sgkf.split(df, df["label"], groups))
    split = pd.Series("train", index=df.index)
    split.iloc[test_idx] = "test"
    return split


def split_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class train/test counts - check that no class has zero test images."""
    rep = df.pivot_table(index="label", columns="split", values="image_id", aggfunc="count", fill_value=0)
    rep["test_frac"] = np.round(rep.get("test", 0) / rep.sum(axis=1), 2)
    return rep
