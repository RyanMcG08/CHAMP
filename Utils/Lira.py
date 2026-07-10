"""
lira.py

LiRA (Likelihood Ratio Attack) style backdoor-detection module.

Given a DataLoader of clean training data and a model architecture, this
trains a set of "shadow" models -- half poisoned with a backdoor trigger
("IN"), half trained clean ("OUT") -- and fits a likelihood-ratio test on
each model's confidence when shown triggered inputs. The resulting
detector can then be pointed at a *new* model to estimate whether it was
trained with a backdoor.

Typical usage
-------------
    from lira import LiraBackdoorDetector, BackdoorConfig, LiraConfig

    def model_fn():
        return MyCNN()

    detector = LiraBackdoorDetector(
        model_fn=model_fn,
        backdoor_cfg=BackdoorConfig(poison_frac=0.1, target_label=0),
        lira_cfg=LiraConfig(n_shadow=32, epochs=15),
    )
    detector.fit(train_loader)                 # trains shadow models

    llr = detector.predict_llr(some_model)      # log-likelihood ratio
    prob = detector.predict_proba(some_model)   # P(model is backdoored)
    is_bd = detector.predict(some_model)        # bool

This is intended for auditing / defensive research (e.g. checking whether
a model you've received or trained has a backdoor), not for tuning an
attack to evade detection.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import random_split, Subset, DataLoader
from torch.utils.data import DataLoader, ConcatDataset


# ---------------------------------------------------------------
# Trigger utilities
# ---------------------------------------------------------------

def default_3x3_x_trigger(x: torch.Tensor, patch_value: float = 1.0,
                           corner: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Stamp a 3x3 'X' pattern into the bottom-right corner of each image.

    x: (N, C, H, W) tensor, values expected in [0, 1].
    Returns a new tensor (input is not modified in place).
    """
    x = x.clone()

    r0, c0 = 1,1

    x[..., r0:r0+3, c0:c0+3] = patch_value

    return x


def poison_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    trigger_fn: Callable[[torch.Tensor], torch.Tensor],
    frac: float,
    source_label: int,
    target_label: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    x = x.clone()
    y = y.clone()

    source_idx = torch.where(y == source_label)[0]

    if len(source_idx) == 0:
        return x, y

    n_poison = max(1, int(len(source_idx) * frac))

    if generator is not None:
        perm = torch.randperm(len(source_idx), generator=generator)
    else:
        perm = torch.randperm(len(source_idx))

    poison_idx = source_idx[perm[:n_poison]]

    x[poison_idx] = trigger_fn(x[poison_idx])
    y[poison_idx] = target_label

    return x, y


# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------

@dataclass
class BackdoorConfig:
    trigger_fn: Callable[[torch.Tensor], torch.Tensor] = default_3x3_x_trigger
    poison_frac: float = 0.1     # fraction of a backdoored shadow model's data that's poisoned
    source_label: int = 0        # label the trigger is meant to force
    target_label: int = 1


@dataclass
class LiraConfig:
    n_shadow: int = 32               # number of shadow models (split ~50/50 IN/OUT)
    shadow_subset_size: int = 2000   # training set size per shadow model
    epochs: int = 15
    lr: float = 0.01
    batch_size: int = 128
    n_query: int = 200               # # of clean images used to probe each model with the trigger
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: Optional[int] = 0


# ---------------------------------------------------------------
# Data / confidence utilities
# ---------------------------------------------------------------

def extract_pool(dataloader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialize all (x, y) pairs from a DataLoader into tensors."""
    xs, ys = [], []
    for xb, yb in dataloader:
        xs.append(xb)
        ys.append(yb)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def logit_scaled_confidence(probs: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p_y = probs[np.arange(len(y)), y]
    p_y = np.clip(p_y, eps, 1 - eps)
    return np.log(p_y / (1 - p_y))


@torch.no_grad()
def get_confidences(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                     device: str, batch_size: int = 512) -> np.ndarray:
    model = model.to(device)
    model.eval()
    all_probs = []
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size].to(device)
        logits = model(xb)
        all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    return logit_scaled_confidence(probs, y.numpy())


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
             device: str, batch_size: int = 512) -> float:
    model = model.to(device)
    model.eval()
    correct = 0
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size].to(device)
        yb = y[i:i + batch_size].to(device)
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
    return correct / len(x)


def train_model(model_fn: Callable[[], nn.Module],
                 train_x: torch.Tensor, train_y: torch.Tensor,
                 cfg: LiraConfig) -> nn.Module:
    model = model_fn().to(cfg.device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
    ds = torch.utils.data.TensorDataset(train_x, train_y)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    return model


# ---------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------

class LiraBackdoorDetector:
    """
    Shadow-model-based backdoor detector using a LiRA-style likelihood
    ratio test.

    fit() trains n_shadow shadow models on data drawn from the pool
    extracted from the supplied DataLoader -- alternating between
    "IN" (backdoor-poisoned) and "OUT" (clean) -- and records each
    model's mean confidence on the target label for triggered inputs.

    predict_llr() / predict_proba() / predict() then score a *new*
    model (e.g. one you want to audit) against the fitted IN/OUT
    confidence distributions.
    """

    def __init__(self, model_fn: Callable[[], nn.Module],
                 backdoor_cfg: Optional[BackdoorConfig] = None,
                 lira_cfg: Optional[LiraConfig] = None):
        self.model_fn = model_fn
        self.backdoor_cfg = backdoor_cfg or BackdoorConfig()
        self.cfg = lira_cfg or LiraConfig()

        self.pool_x: Optional[torch.Tensor] = None
        self.pool_y: Optional[torch.Tensor] = None
        self.query_x: Optional[torch.Tensor] = None
        self.query_y_target: Optional[torch.Tensor] = None

        self.in_scores: Optional[np.ndarray] = None
        self.out_scores: Optional[np.ndarray] = None
        self._fitted = False

    # -- internals --------------------------------------------------

    def _prepare_query_set(self):
        """
        Build a fixed set of clean query images (excluding the target
        label) that will be triggered and used to probe every shadow
        model and every model scored later.
        """
        source_label = self.backdoor_cfg.source_label
        target_label = self.backdoor_cfg.target_label

        mask = self.pool_y == source_label
        idx_pool = torch.nonzero(mask, as_tuple=True)[0]
        n = min(self.cfg.n_query, len(idx_pool))

        if self.cfg.seed is not None:
            g = torch.Generator().manual_seed(self.cfg.seed)
            perm = torch.randperm(len(idx_pool), generator=g)
        else:
            perm = torch.randperm(len(idx_pool))

        sel = idx_pool[perm[:n]]
        self.query_x = self.pool_x[sel].clone()
        self.query_y_target = torch.full((n,), target_label, dtype=torch.long)

    def _train_shadow(self, is_backdoored: bool) -> nn.Module:
        n = len(self.pool_x)
        subset_size = min(self.cfg.shadow_subset_size, n)
        idx = np.random.choice(n, subset_size, replace=False)
        x, y = self.pool_x[idx], self.pool_y[idx]

        if is_backdoored:
            x, y = poison_batch(
                x,
                y,
                self.backdoor_cfg.trigger_fn,
                self.backdoor_cfg.poison_frac,
                self.backdoor_cfg.source_label,
                self.backdoor_cfg.target_label,
            )
        return train_model(self.model_fn, x, y, self.cfg)

    def _trigger_confidence(self, model: nn.Module) -> float:
        """Mean logit-scaled confidence on target_label for triggered query images."""
        triggered = self.backdoor_cfg.trigger_fn(self.query_x)
        conf = get_confidences(model, triggered, self.query_y_target, self.cfg.device)
        return float(conf.mean())

    # -- public API ---------------------------------------------------
    def getOriginal(self, loader,clean_dataset):
        transform = transforms.Compose([transforms.ToTensor()])
        # Re-download the clean dataset
        clean_dataset = clean_dataset("../dataset", train=True, download=True, transform=transform)

        clean_subsets = []

        # Recreate every subset using the same indices
        for subset in loader.dataset.datasets:
            clean_subsets.append(
                Subset(clean_dataset, subset.indices)
            )

        clean_concat = ConcatDataset(clean_subsets)

        return DataLoader(
            clean_concat,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            drop_last=loader.drop_last,
        )
    def fit(self, dataloader,dataset, verbose: bool = True) -> "LiraBackdoorDetector":
        """
        dataloader: a torch DataLoader yielding (x, y) batches of clean
        training data. All data is materialized into an in-memory pool
        used both to build shadow-model training subsets and the fixed
        query set used to probe models with the trigger.
        """
        dataloader = self.getOriginal(dataloader,dataset)
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        self.pool_x, self.pool_y = extract_pool(dataloader)
        self._prepare_query_set()

        in_scores, out_scores = [], []
        for s in range(self.cfg.n_shadow):
            is_backdoored = (s % 2 == 0)
            model = self._train_shadow(is_backdoored)
            conf = self._trigger_confidence(model)
            (in_scores if is_backdoored else out_scores).append(conf)

            if verbose:
                tag = "IN (backdoored)" if is_backdoored else "OUT (clean)"
                print(f"[lira] shadow {s + 1}/{self.cfg.n_shadow} [{tag}] "
                      f"trigger->target confidence: {conf:.3f}")

        self.in_scores = np.array(in_scores)
        self.out_scores = np.array(out_scores)
        self._fitted = True
        return self

    def predict_llr(self, model: nn.Module) -> float:
        """Log-likelihood ratio that `model` is backdoored vs clean."""
        if not self._fitted:
            raise RuntimeError("Call .fit(dataloader) before scoring a model.")

        q = self._trigger_confidence(model)
        mu_in, sigma_in = self.in_scores.mean(), self.in_scores.std() + 1e-6
        mu_out, sigma_out = self.out_scores.mean(), self.out_scores.std() + 1e-6

        ll_in = stats.norm.logpdf(q, mu_in, sigma_in)
        ll_out = stats.norm.logpdf(q, mu_out, sigma_out)
        return float(ll_in - ll_out)

    def predict_proba(self, model: nn.Module) -> float:
        """Rough probability estimate that `model` is backdoored (sigmoid of LLR)."""
        llr = self.predict_llr(model)
        return float(1 / (1 + np.exp(-llr)))

    def predict(self, model: nn.Module, threshold: float = 0.0) -> bool:
        """Boolean call: is `model` backdoored? (LLR > threshold)"""
        return self.predict_llr(model) > threshold

    def evaluate(self, models: List[nn.Module], labels: List[bool]) -> dict:
        """
        Evaluate detector AUC against a labeled set of held-out models.
        labels[i] = True if models[i] is backdoored.
        Requires scikit-learn.
        """
        from sklearn.metrics import roc_auc_score
        scores = [self.predict_llr(m) for m in models]
        auc = roc_auc_score(labels, scores)
        return {"auc": auc, "scores": scores}


# ---------------------------------------------------------------
# Example usage (run directly: python lira.py)
# ---------------------------------------------------------------

if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as T

    def model_fn():
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    transform = T.Compose([T.ToTensor()])
    train_set = torchvision.datasets.MNIST(root="./data", train=True,
                                            download=True, transform=transform)

    # Use a modest subset as the "pool" DataLoader for a quick demo.
    subset = torch.utils.data.Subset(train_set, range(5000))
    train_loader = torch.utils.data.DataLoader(subset, batch_size=256, shuffle=False)

    detector = LiraBackdoorDetector(
        model_fn=model_fn,
        backdoor_cfg=BackdoorConfig(poison_frac=0.1, target_label=0),
        lira_cfg=LiraConfig(n_shadow=16, epochs=10, shadow_subset_size=1500),
    )
    detector.fit(train_loader)

    # Train one known-backdoored and one known-clean model to sanity check.
    print("\n=== Sanity check on two freshly trained models ===")
    bd_model = detector._train_shadow(is_backdoored=True)
    clean_model = detector._train_shadow(is_backdoored=False)

    print(f"Backdoored model  -> LLR={detector.predict_llr(bd_model):.2f}, "
          f"P(backdoored)={detector.predict_proba(bd_model):.3f}")
    print(f"Clean model       -> LLR={detector.predict_llr(clean_model):.2f}, "
          f"P(backdoored)={detector.predict_proba(clean_model):.3f}")