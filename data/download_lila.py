"""Download a small, CPU-friendly subset of LILA BC "WCS Camera Traps".

Steps:
  1. fetch the COCO-style metadata (43 MB zip) once and cache a slim CSV of
     single-label annotations
  2. per class (src/perception/camera_trap/config.yaml): filter countries,
     keep ONE image per burst sequence (near-duplicates inflate accuracy),
     sample up to max_per_class
  3. location-grouped train/test split (no camera in both)
  4. download images in parallel, resized to <=1280 px (MegaDetector input size)

Output: data/lila_wcs/images/<label>/<id>.jpg and data/lila_wcs/manifest.csv
Usage:  .venv/bin/python data/download_lila.py [--max-per-class 150] [--smoke]
        --smoke downloads 2 images/class for the pipeline smoke test.
"""
import argparse
import io
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.paths import LILA_DIR  # noqa: E402
from src.perception.camera_trap.dataset import load_config, location_split, split_report  # noqa: E402

META_URL = "https://storage.googleapis.com/public-datasets-lila/wcs/wcs_camera_traps.json.zip"
IMG_BASE = "https://storage.googleapis.com/public-datasets-lila/wcs-unzipped/"
SLIM_CSV = LILA_DIR / "wcs_single_label_annotations.csv"


def slim_metadata() -> pd.DataFrame:
    """One row per image that has exactly one category label."""
    if SLIM_CSV.exists():
        return pd.read_csv(SLIM_CSV, dtype={"location": str})
    LILA_DIR.mkdir(parents=True, exist_ok=True)
    zpath = LILA_DIR / "wcs_camera_traps.json.zip"
    if not zpath.exists():
        print("downloading WCS metadata (43 MB)...")
        with requests.get(META_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            zpath.write_bytes(r.content)
    print("parsing metadata (800 MB JSON, ~1 min)...")
    with zipfile.ZipFile(zpath) as zf:
        d = json.load(zf.open(zf.namelist()[0]))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    ann = pd.DataFrame(d["annotations"])[["image_id", "category_id"]]
    n_labels = ann.groupby("image_id")["category_id"].nunique()
    ann = ann[ann.image_id.isin(n_labels[n_labels == 1].index)].drop_duplicates("image_id")
    imgs = pd.DataFrame(d["images"])
    imgs = imgs[imgs.get("corrupt", False) != True]  # noqa: E712
    df = ann.merge(imgs, left_on="image_id", right_on="id")
    df["wcs_label"] = df.category_id.map(cats)
    keep = ["image_id", "file_name", "wcs_label", "country_code", "location", "seq_id", "datetime", "width", "height"]
    df = df[keep]
    df.to_csv(SLIM_CSV, index=False)
    return df


def sample(df: pd.DataFrame, cfg: dict, n_override: int | None) -> pd.DataFrame:
    out = []
    groups = [("experiment", cfg["classes"]), ("zero_shot_only", cfg["zero_shot_only"]), ("triage_empty", cfg["triage_empty"])]
    for role, classes in groups:
        for c in classes:
            sub = df[df.wcs_label == c["wcs_label"]]
            if c.get("countries"):
                sub = sub[sub.country_code.isin(c["countries"])]
            sub = sub.drop_duplicates("seq_id")  # one frame per burst
            n = n_override or c.get("n") or cfg["max_per_class"]
            sub = sub.sample(n=min(n, len(sub)), random_state=cfg["seed"])
            out.append(sub.assign(label=c["key"], role=role))
            print(f"{c['key']:18s} available(1/seq)={len(df[df.wcs_label == c['wcs_label']].drop_duplicates('seq_id')):5d}  sampled={len(sub)}")
    return pd.concat(out, ignore_index=True)


def fetch(row, max_side: int) -> str | None:
    dest = LILA_DIR / "images" / row.label / f"{row.image_id}.jpg"
    if dest.exists():
        return str(dest)
    try:
        r = requests.get(IMG_BASE + row.file_name, timeout=60)
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        im.thumbnail((max_side, max_side))
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, quality=90)
        return str(dest)
    except Exception as e:  # keep going; report failures at the end
        print(f"  failed {row.file_name}: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="2 images per class")
    args = ap.parse_args()
    cfg = load_config()
    n = 2 if args.smoke else args.max_per_class

    df = sample(slim_metadata(), cfg, n)
    exp = df.role == "experiment"
    df["split"] = "zero_shot_only"
    df.loc[df.role == "triage_empty", "split"] = "triage_only"
    if not args.smoke:
        df.loc[exp, "split"] = location_split(df[exp], cfg["test_fraction"], cfg["seed"])

    with ThreadPoolExecutor(max_workers=8) as ex:
        paths = list(tqdm(ex.map(lambda r: fetch(r, cfg["resize_max_side"]), df.itertuples()), total=len(df)))
    df["local_path"] = paths
    failed = df.local_path.isna().sum()
    df = df.dropna(subset=["local_path"])

    name = "manifest_smoke.csv" if args.smoke else "manifest.csv"
    df.to_csv(LILA_DIR / name, index=False)
    print(f"\nwrote {LILA_DIR / name}: {len(df)} images ({failed} failed)")
    if not args.smoke:
        print(split_report(df[df.role == "experiment"]))


if __name__ == "__main__":
    main()
