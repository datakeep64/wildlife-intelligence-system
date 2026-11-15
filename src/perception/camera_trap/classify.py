"""Stage 3 - species classification on frozen BioCLIP embeddings.

Three classifiers, same features:
  zero_shot    : no training data; nearest text prompt in embedding space
  LinearProbe  : logistic regression on image embeddings (learns a boundary)
  KNNProbe     : k-nearest neighbours (no boundary; memory-based)
Comparing them separates "is the representation good?" from "is the label
space (the text prompts) well aligned with my data?".
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier

LOOKALIKE_PAIRS = [
    ("tiger", "leopard"),
    ("tiger", "clouded_leopard"),
    ("leopard", "clouded_leopard"),
    ("sun_bear", "wild_boar"),
    ("sambar", "red_muntjac"),
]


def zero_shot(img_emb: np.ndarray, text_emb: np.ndarray, labels: list[str], temperature: float = 100.0):
    """Return (pred_labels, probs[n, k]) via softmax over cosine similarities."""
    logits = temperature * img_emb @ text_emb.T
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return np.array(labels)[probs.argmax(axis=1)], probs


class LinearProbe:
    def __init__(self, C: float = 1.0):
        self.clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


class KNNProbe:
    def __init__(self, k: int = 5):
        self.clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)


def evaluate(y_true, y_pred, labels: list[str]) -> dict:
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "per_class": {
            lab: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4), "f1": round(float(f[i]), 4), "support": int(n[i])}
            for i, lab in enumerate(labels)
        },
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "lookalike_confusion": lookalike_confusion(cm, labels),
    }


def lookalike_confusion(cm: np.ndarray, labels: list[str]) -> dict:
    """For each look-alike pair (a, b): share of true-a predicted as b, and vice versa."""
    idx = {l: i for i, l in enumerate(labels)}
    out = {}
    for a, b in LOOKALIKE_PAIRS:
        if a in idx and b in idx:
            ia, ib = idx[a], idx[b]
            ra, rb = cm[ia].sum(), cm[ib].sum()
            out[f"{a}->{b}"] = round(float(cm[ia, ib] / ra), 4) if ra else None
            out[f"{b}->{a}"] = round(float(cm[ib, ia] / rb), 4) if rb else None
    return out
