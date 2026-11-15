"""Knowledge-graph embeddings (PyKEEN): TransE vs RotatE on the occurrence KG.

TransE  : a relation is a translation,  h + r ~ t.  It cannot model a symmetric relation
          such as adjacentTo (it would need r = 0), nor one-to-many links well.
RotatE  : a relation is a rotation in complex space, h * r ~ t (|r| = 1). A rotation by pi is
          its own inverse, so symmetric relations are representable.

Both are trained on CPU (the graph is small) and scored only on species -recordedIn-> square.
"""
import numpy as np
import torch
from pykeen.losses import MarginRankingLoss, NSSALoss
from pykeen.models import RotatE, TransE
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory

from .link_prediction import LPSplit

MODELS = {"TransE": TransE, "RotatE": RotatE}
LOSSES = {"margin": lambda: MarginRankingLoss(margin=1.0),
          "self_adversarial": lambda: NSSALoss(margin=9.0, adversarial_temperature=1.0)}


class KGE:
    def __init__(self, model: str, triples: np.ndarray, dim: int = 64, lr: float = 0.005,
                 loss: str = "self_adversarial", num_negs: int = 32, batch_size: int = 512, seed: int = 0):
        torch.manual_seed(seed)
        self.tf = TriplesFactory.from_labeled_triples(triples, create_inverse_triples=False)
        self.model = MODELS[model](triples_factory=self.tf, embedding_dim=dim, loss=LOSSES[loss](),
                                   random_seed=seed).to("cpu")
        self.loop = SLCWATrainingLoop(
            model=self.model, triples_factory=self.tf,
            optimizer=torch.optim.Adam(self.model.get_grad_params(), lr=lr),
            negative_sampler="basic", negative_sampler_kwargs={"num_negs_per_pos": num_negs})
        self.batch_size, self.epochs = batch_size, 0

    def train_to(self, epochs: int) -> "KGE":
        """Train until `epochs` total (continues from the current epoch)."""
        if epochs > self.epochs:
            self.loop.train(triples_factory=self.tf, num_epochs=epochs, batch_size=self.batch_size,
                            continue_training=self.epochs > 0, use_tqdm=False)
            self.epochs = epochs
        return self

    @torch.no_grad()
    def scores(self, split: LPSplit) -> np.ndarray:
        """[n_species, n_cells] plausibility of species -recordedIn-> square."""
        self.model.eval()
        e, r = self.tf.entity_to_id, self.tf.relation_to_id["recordedIn"]
        hr = torch.tensor([[e[s], r] for s in split.species])
        cols = torch.tensor([e[c] for c in split.cells])
        out = torch.cat([self.model.score_t(hr_batch=hr[i:i + 256])[:, cols] for i in range(0, len(hr), 256)])
        self.model.train()
        return out.numpy()
