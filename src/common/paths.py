"""Central paths so every component reads/writes the same places."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
KG_DIR = ROOT / "src" / "reasoning" / "kg"

# raw downloads (gitignored)
LILA_DIR = DATA / "lila_wcs"
GBIF_DIR = DATA / "gbif"
IUCN_CSV = DATA / "iucn_status.csv"

# curated, committed inputs
SPECIES_CSV = KG_DIR / "species.csv"
ZONES_CSV = KG_DIR / "zones.csv"
ONTOLOGY_TTL = KG_DIR / "ontology.ttl"
SEED_TTL = KG_DIR / "species_seed.ttl"


def device() -> str:
    """Pick the best available torch device: Apple MPS, else CPU (CUDA if present)."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
