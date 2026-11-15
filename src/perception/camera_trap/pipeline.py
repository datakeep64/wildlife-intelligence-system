"""End-to-end camera-trap triage: image -> MegaDetector -> crop -> BioCLIP -> sighting.

Each output record is a *sighting* - the hand-off object from perception (PRS)
to reasoning (IRS). Its fields map onto wis:Sighting in the ontology.

Usage:
  .venv/bin/python -m src.perception.camera_trap.pipeline IMG [IMG ...] [--camera CT01 --zone taman_negara]
"""
import argparse
import json
from datetime import datetime, timezone

from PIL import Image

from src.perception.camera_trap.classify import zero_shot
from src.perception.camera_trap.dataset import load_config
from src.perception.camera_trap.detect import Detector, best_box, crop
from src.perception.camera_trap.embed import BioCLIP


class TriagePipeline:
    def __init__(self, cfg: dict | None = None, include_zero_shot_only: bool = True):
        self.cfg = cfg or load_config()
        self.detector = Detector(self.cfg["md_model"], self.cfg["md_threshold"])
        self.clip = BioCLIP(self.cfg["model"])
        keys = [c["key"] for c in self.cfg["classes"]]
        if include_zero_shot_only:
            keys += [c["key"] for c in self.cfg["zero_shot_only"]]
        self.labels = [k for k in keys if k != "human"]  # people are handled by MegaDetector
        self.text_emb = self.clip.text_embeddings([self.cfg["prompts"][k] for k in self.labels])

    def run(self, path: str, camera: str | None = None, zone: str | None = None) -> dict:
        det = self.detector.detect(path)
        sighting = {
            "image": path,
            "camera_id": camera,
            "zone": zone,
            "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "triage": det["triage"],
            "n_animals": sum(d["category"] == "animal" for d in det["detections"]),
            "n_persons": sum(d["category"] == "person" for d in det["detections"]),
            "n_vehicles": sum(d["category"] == "vehicle" for d in det["detections"]),
            "species": None,
            "species_conf": None,
            "top3": None,
            "detections": det["detections"],
        }
        box = best_box(det["detections"], "animal")
        if box is not None:
            emb = self.clip.image_embeddings([crop(Image.open(path), box["bbox"])])
            _, probs = zero_shot(emb, self.text_emb, self.labels)
            order = probs[0].argsort()[::-1][:3]
            sighting["species"] = self.labels[order[0]]
            sighting["species_conf"] = round(float(probs[0][order[0]]), 4)
            sighting["top3"] = [(self.labels[i], round(float(probs[0][i]), 4)) for i in order]
        return sighting


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--camera")
    ap.add_argument("--zone")
    args = ap.parse_args()
    pipe = TriagePipeline()
    for p in args.images:
        print(json.dumps(pipe.run(p, args.camera, args.zone), ensure_ascii=False))


if __name__ == "__main__":
    main()
