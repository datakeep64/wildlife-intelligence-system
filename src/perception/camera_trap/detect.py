"""Stage 1 - triage with MegaDetector v5a: animal / person / vehicle / empty.

MegaDetector is class-agnostic about species: it only finds *where* an animal,
person or vehicle is. That is exactly what triage needs: drop empty frames
(most of a camera-trap archive) and surface person/vehicle frames as a
possible intrusion signal for anti-poaching.
"""
from pathlib import Path

from PIL import Image

CATEGORY = {"1": "animal", "2": "person", "3": "vehicle"}


def triage_label(dets: list[dict]) -> str:
    """One label per frame. Intrusion (person/vehicle) takes precedence over animal."""
    cats = {d["category"] for d in dets}
    for c in ("person", "vehicle", "animal"):
        if c in cats:
            return c
    return "empty"


class Detector:
    def __init__(self, model: str = "MDV5A", threshold: float = 0.2):
        from megadetector.detection.run_detector import load_detector

        self.model = load_detector(model)  # downloads weights (~280 MB) on first use
        self.threshold = threshold

    def detect(self, path: str | Path) -> dict:
        from megadetector.visualization import visualization_utils as vu

        img = vu.load_image(str(path))
        r = self.model.generate_detections_one_image(img, image_id=str(path), detection_threshold=self.threshold)
        dets = [
            {"category": CATEGORY[d["category"]], "conf": round(float(d["conf"]), 4), "bbox": [round(v, 4) for v in d["bbox"]]}
            for d in r["detections"]
        ]
        dets.sort(key=lambda d: -d["conf"])
        return {"image": str(path), "detections": dets, "triage": triage_label(dets)}


def best_box(dets: list[dict], category: str) -> dict | None:
    cands = [d for d in dets if d["category"] == category]
    return max(cands, key=lambda d: d["conf"]) if cands else None


def crop(img: Image.Image, bbox: list[float], pad: float = 0.15, min_side: int = 64) -> Image.Image:
    """Crop a normalised [x, y, w, h] box with padding, expanded to a square.

    Square crops avoid the distortion CLIP-style preprocessing would add when it
    resizes a long thin box; padding keeps some context (tail, legs).
    """
    W, H = img.size
    x, y, w, h = bbox[0] * W, bbox[1] * H, bbox[2] * W, bbox[3] * H
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h, min_side) * (1 + 2 * pad)
    left, top = max(0, cx - side / 2), max(0, cy - side / 2)
    right, bottom = min(W, cx + side / 2), min(H, cy + side / 2)
    return img.crop((int(left), int(top), int(right), int(bottom)))
