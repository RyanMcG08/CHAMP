"""
lira.py

LiRA (Likelihood Ratio Attack) style backdoor-detection module.

Given a DataLoader of clean training data and a model architecture, this
trains a set of "shadow" models -- half poisoned with a backdoor trigger
("IN"), half trained clean ("OUT") -- and fits a likelihood-ratio test on
each model's confidence when shown triggered inputs. The resulting
detector can then be pointed at a *new* model to estimate whether it was
trained with a backdoor.

This is intended for auditing / defensive research (e.g. checking whether
a model you've received or trained has a backdoor), not for tuning an
attack to evade detection.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

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
    Stamp a 3x3 solid patch into the image.

    x: (N, C, H, W) tensor, values expected in [0, 1].
    Returns a new tensor (input is not modified in place).
    """
    x = x.clone()

    r0, c0 = 1, 1

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
    # Either a single fraction applied to every "IN" shadow model, or a
    # sequence of fractions that get distributed round-robin across the
    # IN shadow models (e.g. [0.2, 0.3, 0.5] with 15 IN shadows gives
    # 5 shadows trained at each fraction).
    poison_frac: Union[float, Sequence[float]] = 0.1
    source_label: int = 0        # label whose samples get poisoned
    target_label: int = 1        # label the trigger is meant to force


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
        # Parallel to in_scores: which poison_frac produced each IN score.
        # Useful for inspecting how detectability varies with poison rate.
        self.in_fracs_used: Optional[np.ndarray] = None
        self._fitted = False

    # -- internals --------------------------------------------------

    def _prepare_query_set(self):
        """
        Build a fixed set of clean query images (the source-label class)
        that will be triggered and used to probe every shadow model and
        every model scored later.
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

    def _poison_frac_schedule(self, n_backdoored: int) -> List[float]:
        """
        Build the list of poison fractions to apply across the
        n_backdoored "IN" shadow models.

        If backdoor_cfg.poison_frac is a single float, every IN shadow
        gets that fraction (original behavior).

        If it's a sequence, e.g. [0.2, 0.3, 0.5], the IN shadow models
        are assigned fractions round-robin so each value gets used as
        evenly as possible -- with n_backdoored=15 and 3 fractions,
        that's 5 shadows per fraction; with 16 and 3 fractions, it's
        6/5/5.
        """
        pf = self.backdoor_cfg.poison_frac
        if isinstance(pf, (list, tuple, np.ndarray)):
            fracs = list(pf)
        else:
            fracs = [pf]
        return [fracs[i % len(fracs)] for i in range(n_backdoored)]

    def _default_poison_frac(self) -> float:
        """Fallback fraction used when _train_shadow(True) is called directly
        without an explicit poison_frac (e.g. ad-hoc sanity checks)."""
        pf = self.backdoor_cfg.poison_frac
        if isinstance(pf, (list, tuple, np.ndarray)):
            return float(pf[0])
        return float(pf)

    def _fit_in_distributions(self) -> dict:
        """
        Fit one Gaussian per distinct poison_frac used among the IN
        (backdoored) shadow models, rather than pooling all IN scores
        into a single Gaussian.

        Per-fraction means are used as-is, but the standard deviation is
        pooled ("global variance" trick) across all fractions -- each
        group's scores are centered on its own mean, then all residuals
        are combined into one std estimate. This matters because with
        e.g. 16 IN shadows split across 3 fractions, each group only has
        ~5 samples, too few to estimate a per-group std reliably on its
        own.

        Returns: dict mapping poison_frac -> (mu, sigma)
        """
        fracs = self.in_fracs_used
        scores = self.in_scores

        if fracs is None or len(fracs) == 0:
            raise RuntimeError("No IN shadow scores available; call fit() first.")

        unique_fracs = np.unique(fracs)

        means = {}
        residuals = []
        for f in unique_fracs:
            group = scores[fracs == f]
            mu = float(group.mean())
            means[f] = mu
            if len(group) > 1:
                residuals.append(group - mu)

        if residuals:
            pooled = np.concatenate(residuals)
            global_sigma_in = float(pooled.std() + 1e-6)
        else:
            # Fallback: not enough samples anywhere to pool residuals
            global_sigma_in = float(scores.std() + 1e-6)

        return {float(f): (means[f], global_sigma_in) for f in unique_fracs}

    def _train_shadow(self, is_backdoored: bool,
                       poison_frac: Optional[float] = None) -> nn.Module:
        n = len(self.pool_x)
        subset_size = min(self.cfg.shadow_subset_size, n)
        idx = np.random.choice(n, subset_size, replace=False)
        x, y = self.pool_x[idx], self.pool_y[idx]

        if is_backdoored:
            frac = poison_frac if poison_frac is not None else self._default_poison_frac()
            x, y = poison_batch(
                x,
                y,
                self.backdoor_cfg.trigger_fn,
                frac,
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
    def getOriginal(self, loader, clean_dataset):
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

    def fit(self, dataloader, dataset, verbose: bool = True) -> "LiraBackdoorDetector":
        """
        dataloader: a torch DataLoader over the (possibly backdoored)
        malicious-client samples. Its underlying dataset must be a
        ConcatDataset of Subsets so the original indices can be
        recovered against `dataset`.
        dataset: the original torchvision dataset class (e.g.
        torchvision.datasets.MNIST) used to look up unpoisoned
        originals of the same samples.
        """
        dataloader = self.getOriginal(dataloader, dataset)
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        self.pool_x, self.pool_y = extract_pool(dataloader)
        self._prepare_query_set()

        # How many shadow models will be "IN" (backdoored) given the
        # alternating s % 2 == 0 assignment below.
        n_backdoored = sum(1 for s in range(self.cfg.n_shadow) if s % 2 == 0)
        frac_schedule = self._poison_frac_schedule(n_backdoored)

        in_scores, out_scores = [], []
        in_fracs_used = []
        bd_counter = 0

        for s in range(self.cfg.n_shadow):
            is_backdoored = (s % 2 == 0)

            if is_backdoored:
                frac = frac_schedule[bd_counter]
                bd_counter += 1
                model = self._train_shadow(True, poison_frac=frac)
                conf = self._trigger_confidence(model)
                in_scores.append(conf)
                in_fracs_used.append(frac)
                if verbose:
                    print(f"[lira] shadow {s + 1}/{self.cfg.n_shadow} "
                          f"[IN backdoored, poison_frac={frac}] "
                          f"trigger->target confidence: {conf:.3f}")
            else:
                model = self._train_shadow(False)
                conf = self._trigger_confidence(model)
                out_scores.append(conf)
                if verbose:
                    print(f"[lira] shadow {s + 1}/{self.cfg.n_shadow} "
                          f"[OUT clean] trigger->target confidence: {conf:.3f}")

        self.in_scores = np.array(in_scores)
        self.out_scores = np.array(out_scores)
        self.in_fracs_used = np.array(in_fracs_used)
        self._fitted = True
        return self

    def state_dict(self) -> dict:
        """
        Everything needed to reconstruct a fitted detector, minus the
        shadow models themselves (which aren't kept around after fit()
        anyway -- only their derived confidence scores are).
        """
        return {
            "in_scores": self.in_scores,
            "out_scores": self.out_scores,
            "in_fracs_used": self.in_fracs_used,
            "query_x": self.query_x,
            "query_y_target": self.query_y_target,
            "backdoor_cfg": self.backdoor_cfg,
            "cfg": self.cfg,
            "_fitted": self._fitted,
        }

    def load_state_dict(self, state: dict) -> None:
        self.in_scores = state["in_scores"]
        self.out_scores = state["out_scores"]
        self.in_fracs_used = state["in_fracs_used"]
        self.query_x = state["query_x"]
        self.query_y_target = state["query_y_target"]
        self.backdoor_cfg = state["backdoor_cfg"]
        self.cfg = state["cfg"]
        self._fitted = state["_fitted"]

    def save(self, path: str) -> None:
        """Save the fitted detector state to disk (via torch.save)."""
        if not self._fitted:
            raise RuntimeError("Detector isn't fitted yet -- call fit() before save().")
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, model_fn: Callable[[], nn.Module]) -> "LiraBackdoorDetector":
        """
        Reload a previously saved detector. `model_fn` must be supplied
        again since it's a live callable (not something we saved) --
        pass the same architecture factory used at fit() time so
        predict_llr() etc. can be called on new models of that type.

        Note: torch.load with weights_only=False will unpickle arbitrary
        objects (including backdoor_cfg.trigger_fn). Only load files you
        saved yourself / trust.
        """
        state = torch.load(path, weights_only=False)
        detector = cls(
            model_fn=model_fn,
            backdoor_cfg=state["backdoor_cfg"],
            lira_cfg=state["cfg"],
        )
        detector.load_state_dict(state)
        return detector

    def predict_llr(self, model: nn.Module) -> float:
        """Log-likelihood ratio that `model` is backdoored vs clean."""
        if not self._fitted:
            raise RuntimeError("Call .fit(dataloader, dataset) before scoring a model.")

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

    def predict_llr_stratified(self, model: nn.Module) -> Tuple[float, float]:
        """
        Score a model against each per-poison_frac IN distribution
        separately (rather than one pooled IN Gaussian), and return the
        best (max) log-likelihood ratio vs. the OUT distribution, along
        with which poison_frac it best matches.

        This is the recommended scoring method when
        BackdoorConfig.poison_frac was a list/range during fit(): a
        single pooled Gaussian over IN scores spanning several poison
        rates can end up wide or even bimodal, which weakens the LLR
        test. Fitting one Gaussian per fraction and taking the best
        match avoids that, at the cost of not knowing in advance which
        fraction (if any) the query model used -- we just report
        whichever hypothesis explains the observed confidence best.

        Returns: (best_llr, best_matching_poison_frac)
        """
        if not self._fitted:
            raise RuntimeError("Call .fit(dataloader, dataset) before scoring a model.")

        in_dists = self._fit_in_distributions()
        mu_out, sigma_out = self.out_scores.mean(), self.out_scores.std() + 1e-6

        q = self._trigger_confidence(model)
        ll_out = stats.norm.logpdf(q, mu_out, sigma_out)

        best_llr = -np.inf
        best_frac = None
        for frac, (mu_in, sigma_in) in in_dists.items():
            ll_in = stats.norm.logpdf(q, mu_in, sigma_in)
            llr = ll_in - ll_out
            if llr > best_llr:
                best_llr = llr
                best_frac = frac

        return float(best_llr), float(best_frac)

    def predict_proba_stratified(self, model: nn.Module) -> float:
        """Sigmoid of the stratified (best-fraction-match) LLR."""
        llr, _ = self.predict_llr_stratified(model)
        return float(1 / (1 + np.exp(-llr)))

    def predict_stratified(self, model: nn.Module, threshold: float = 0.0) -> bool:
        """Boolean call using the stratified LLR: is `model` backdoored?"""
        llr, _ = self.predict_llr_stratified(model)
        return llr > threshold

    def evaluate(self, models: List[nn.Module], labels: List[bool],
                 stratified: bool = False) -> dict:
        """
        Evaluate detector AUC against a labeled set of held-out models.
        labels[i] = True if models[i] is backdoored.

        stratified=True uses predict_llr_stratified (per-fraction
        Gaussians, best match) instead of the pooled predict_llr --
        generally the better choice when poison_frac was a range during
        fit(). Requires scikit-learn.
        """
        from sklearn.metrics import roc_auc_score

        if stratified:
            results = [self.predict_llr_stratified(m) for m in models]
            scores = [llr for llr, _ in results]
            fracs = [frac for _, frac in results]
            auc = roc_auc_score(labels, scores)
            return {"auc": auc, "scores": scores, "best_matching_fracs": fracs}
        else:
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

    # Example: split IN shadow models evenly across three poison rates.
    detector = LiraBackdoorDetector(
        model_fn=model_fn,
        backdoor_cfg=BackdoorConfig(poison_frac=[0.2, 0.3, 0.5],
                                     source_label=0, target_label=1),
        lira_cfg=LiraConfig(n_shadow=16, epochs=10, shadow_subset_size=1500),
    )

    # detector.fit(malicious_loader, torchvision.datasets.MNIST)
    #
    # After fitting, detector.in_fracs_used lines up with detector.in_scores
    # so you can inspect how confidence separation changes with poison_frac.
    #
    # Scoring a new model:
    #   detector.predict_llr(model)              # pooled IN Gaussian (all fracs mixed)
    #   detector.predict_llr_stratified(model)    # per-fraction Gaussians, best match
    #       -> (llr, best_matching_poison_frac)
    #   detector.evaluate(models, labels, stratified=True)  # AUC using stratified scoring