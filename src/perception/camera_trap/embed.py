"""Stage 2 - BioCLIP image/text embeddings (frozen; no fine-tuning).

BioCLIP is a CLIP model trained on the Tree of Life (TreeOfLife-10M), aligning
images with taxonomic names. One frozen encoder gives us both:
  - zero-shot classification (compare image vs text embeddings), and
  - features for a linear probe / kNN (the transfer-learning experiment).
"""
import numpy as np
import torch
from PIL import Image

from src.common.paths import device as best_device


class BioCLIP:
    def __init__(self, model_name: str = "hf-hub:imageomics/bioclip", device: str | None = None):
        import open_clip

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device or best_device()
        try:
            self.model.to(self.device).eval()
        except RuntimeError:  # some ops unsupported on MPS -> CPU fallback
            self.device = "cpu"
            self.model.to("cpu").eval()

    @torch.no_grad()
    def image_embeddings(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([self.preprocess(im.convert("RGB")) for im in images[i : i + batch_size]])
            f = self.model.encode_image(batch.to(self.device))
            out.append(torch.nn.functional.normalize(f, dim=-1).cpu().float().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), dtype=np.float32)

    @torch.no_grad()
    def text_embeddings(self, prompts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(prompts).to(self.device)
        f = self.model.encode_text(tokens)
        return torch.nn.functional.normalize(f, dim=-1).cpu().float().numpy()
