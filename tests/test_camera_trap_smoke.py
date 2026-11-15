"""Smoke test: a few real WCS images through MegaDetector + BioCLIP.

Needs data/lila_wcs/manifest_smoke.csv (python data/download_lila.py --smoke).
First run downloads MegaDetector (~280 MB) and BioCLIP (~600 MB) weights.
Run: .venv/bin/python -m pytest tests/test_camera_trap_smoke.py -s
"""
import pandas as pd
import pytest

from src.common.paths import LILA_DIR

MANIFEST = LILA_DIR / "manifest_smoke.csv"


@pytest.fixture(scope="module")
def pipe():
    from src.perception.camera_trap.pipeline import TriagePipeline

    return TriagePipeline()


@pytest.mark.skipif(not MANIFEST.exists(), reason="run data/download_lila.py --smoke first")
@pytest.mark.parametrize("label", ["tiger", "sun_bear", "human", "empty"])
def test_pipeline_on_real_images(pipe, label):
    m = pd.read_csv(MANIFEST)
    if not (m.label == label).any():
        pytest.skip(f"no '{label}' images (LILA withholds human images for privacy)")
    row = m[m.label == label].iloc[0]
    s = pipe.run(row.local_path, camera="smoke", zone="taman_negara")
    print(f"\n[{label}] triage={s['triage']} species={s['species']} conf={s['species_conf']} top3={s['top3']}")
    assert s["triage"] in {"animal", "person", "vehicle", "empty"}
    if s["triage"] == "animal" or s["n_animals"]:
        assert s["species"] in pipe.labels
        assert 0.0 <= s["species_conf"] <= 1.0


def test_zero_shot_math():
    import numpy as np

    from src.perception.camera_trap.classify import zero_shot

    img = np.array([[1.0, 0.0], [0.0, 1.0]])
    txt = np.array([[1.0, 0.0], [0.0, 1.0]])
    pred, probs = zero_shot(img, txt, ["a", "b"])
    assert list(pred) == ["a", "b"] and np.allclose(probs.sum(1), 1)
