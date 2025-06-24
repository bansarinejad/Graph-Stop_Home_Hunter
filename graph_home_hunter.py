#!/usr/bin/env python
"""
Graph-Stop Home Hunter

A hybrid graph-and-tabular pipeline for recommending when a buyer
should **place an offer** on a property—or hold off for a potentially
better listing next week.

Main components
---------------
1. **ListingGraphTG** – converts a weekly CSV into a PyTorch-Geometric
   hetero-type graph (buyer ▸ listing ▸ suburb).
2. **GCN** – lightweight 2-layer GAT encoder that produces
   128-dim embeddings for each node.
3. **IntentHead** – Bayesian or Logistic head on top of
   `[embeddings | one-hot tabular]` with optional mutual-information
   feature selection and calibrated probabilities.
4. **OptionValuePlanner (OVP)** – Bayesian stop-or-wait heuristic that
   compares expected regret of buying now vs. waiting one week.
5. **CLI** – sub-commands for training, sweeping thresholds,
   prediction, and scripted week-by-week simulations.

Only comments / docstrings have been added; *functional behaviour is
unchanged.*
"""
# ────────────────────────────────────────────────────────────────
#                     STANDARD LIBRARY IMPORTS
# ────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ────────────────────────────────────────────────────────────────
#                     THIRD-PARTY IMPORTS
# ────────────────────────────────────────────────────────────────
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv

# ────────────────────────────────────────────────────────────────
#                  GLOBAL REPRODUCIBILITY HELPER
# ────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Force deterministic behaviour across Python, NumPy and PyTorch."""
    import numpy as _np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ────────────────────────────────────────────────────────────────
#              TORCH LOAD SHIM (for newer checkpoints)
# ────────────────────────────────────────────────────────────────
def _safe_torch_load(path: Path):
    """
    Torch >= 2.2 introduces *weights_only* checkpoints that raise a
    `TypeError` on older load calls. This helper retries gracefully.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # fallback for weight-only files
        return torch.load(path, map_location="cpu")


# Streamlit sometimes imports private torch modules; create dummies up-front
for mod in ("classes", "_classes"):
    if f"torch.{mod}" not in sys.modules:
        fake = types.ModuleType(f"torch.{mod}")
        fake.__path__ = types.SimpleNamespace(_path=[])
        setattr(torch, mod, fake)
        sys.modules[f"torch.{mod}"] = fake


# ────────────────────────────────────────────────────────────────
#     TINY NUMERIC VECTOR – 6 NORMALISED FEATURES PER LISTING
# ────────────────────────────────────────────────────────────────
def vec(r: Dict[str, Any]) -> List[float]:
    """Create a fixed-length numeric feature vector from a CSV row."""
    return [
        r["asking_price"] / 1e6,
        r["body_corp"] / 1e4,
        r["beds"] / 5,
        r["internal_m2"] / 150,
        r["travel_time_mins"] / 60,
        r["livability"] / 100,
    ]


# ────────────────────────────────────────────────────────────────
#                   USER-PREFERENCE SCORING
# ────────────────────────────────────────────────────────────────
from dataclasses import dataclass, field


@dataclass
class UserPreferences:
    """Hard/soft constraints that define a 'desirable' listing."""
    max_price: float = 600_000
    max_body_corp: float = 6_000
    min_beds: int = 2
    min_internal_m2: float = 50
    good_condition: bool = True
    no_cladding: bool = True
    max_travel_mins: int = 45
    near_shops: bool = True
    min_livability: float = 0.0
    # Optional booleans – `None` means 'don't care'
    has_parking: Optional[bool] = None
    has_storage: Optional[bool] = None
    has_solar: Optional[bool] = None
    preferred_areas: List[str] = field(
        default_factory=lambda: [
            "Melbourne CBD",
            "Brunswick",
            "St Kilda",
            "Carlton",
            "Hawthorn",
        ]
    )
    north_facing: Optional[bool] = None
    outdoor_space: Optional[bool] = None
    # Soft-score weights
    w_top: float = 1.0
    w_lower: float = 0.6
    w_lowest: float = 0.3


class PreferenceScorer:
    """Rule-based label generator used for *supervised* GNN training."""

    def __init__(self, prefs: UserPreferences) -> None:
        self.p = prefs

    # ---------------- internal helpers ----------------
    @staticmethod
    def _b(val):
        """Convert common truthy/falsey strings to bool."""
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            m = {
                "yes": 1,
                "true": 1,
                "y": 1,
                "1": 1,
                "no": 0,
                "false": 0,
                "n": 0,
                "0": 0,
            }
            return {1: True, 0: False}.get(m.get(val.strip().lower()))

    def _match(self, val, pref) -> bool:
        """Return `True` if preference is *None* or values match."""
        return True if pref is None else self._b(val) == pref

    # ---------------- public API ----------------
    def score(self, r: Dict[str, Any]) -> float:
        """
        Returns a **soft score** (≥ 0) if the listing satisfies all hard
        constraints, otherwise ``-inf``.

        That soft score is later thresholded into a binary label.
        """
        p = self.p
        t = r.get("travel_time_mins", 60)

        # -------- hard constraints --------
        hard_ok = (
            r["asking_price"] <= p.max_price
            and r["body_corp"] <= p.max_body_corp
            and r["beds"] >= p.min_beds
            and r["internal_m2"] >= p.min_internal_m2
            and (not p.good_condition or r["condition"] in ("good", "new"))
            and (not p.no_cladding or not r["has_cladding"])
            and t <= p.max_travel_mins
            and (not p.near_shops or r["near_shops"])
            and r["livability"] >= p.min_livability
        )
        if not hard_ok:
            return -math.inf

        # -------- soft extras --------
        s = p.w_top
        if self._match(r["has_parking"], p.has_parking):
            s += p.w_lower
        if self._match(r["has_storage"], p.has_storage):
            s += p.w_lower
        if self._match(r["has_solar"], p.has_solar):
            s += p.w_lower
        if r["suburb"] in p.preferred_areas:
            s += p.w_lower
        if self._match(r["north_facing"], p.north_facing):
            s += p.w_lowest
        if self._match(r["outdoor_space"], p.outdoor_space):
            s += p.w_lowest
        return s


# ────────────────────────────────────────────────────────────────
#                 GRAPH CONSTRUCTION (PyG Data)
# ────────────────────────────────────────────────────────────────
class ListingGraphTG:
    """
    Build a homogeneous PyG **Data** object with three conceptual
    node-types:

    * Node 0 – synthetic *buyer* (no real features)
    * Nodes 1…N – listings for the current week
    * Nodes N+1… – one node per suburb centroid

    Edges:
        • buyer → listing            (ranking supervision)
        • listing → suburb-centroid  (spatial context)

    ``g.listing_idx`` holds indices of listing nodes for convenience.
    """

    def __init__(self, prefs: UserPreferences | None = None) -> None:
        self.scorer = PreferenceScorer(prefs or UserPreferences())

    # ---------------- load helpers ----------------
    def from_csv(self, csv: Path, add_masks: bool = True) -> Data:
        """Convenience wrapper that reads a CSV then calls *from_df*."""
        return self.from_df(pd.read_csv(csv), add_masks)

    # ---------------- main builder ----------------
    def from_df(self, df: pd.DataFrame, add_masks: bool = True) -> Data:
        # Feature / label holders
        buyer_feat = torch.zeros(8)  # placeholder vector
        listing_feats, suburb_feats, e_src, e_dst, labels = [], [], [], [], []
        suburb_to_id: Dict[str, int] = {}

        # -------- iterate over listings --------
        for i, row in df.iterrows():
            lid = 1 + i  # listing node id
            listing_feats.append(torch.tensor(vec(row) + [0, 0]))
            labels.append(0 if self.scorer.score(row) == -math.inf else 1)

            # buyer → listing edge
            e_src.append(0)
            e_dst.append(lid)

            # listing → suburb edge (create suburb node lazily)
            sb = row["suburb"]
            if sb not in suburb_to_id:
                sb_id = 1 + len(df) + len(suburb_to_id)
                suburb_to_id[sb] = sb_id
                f = torch.zeros(8)
                f[5] = row["livability"] / 100  # inject livability
                f[6:] = torch.tensor([0, 1])  # one-hot: suburb flag
                suburb_feats.append((sb_id, f))
            e_src.append(lid)
            e_dst.append(suburb_to_id[sb])

        # -------- stack tensors & annotate graph --------
        x = torch.vstack([buyer_feat] + listing_feats + [f for _, f in suburb_feats])
        g = Data(x=x, edge_index=torch.tensor([e_src, e_dst], dtype=torch.long))
        g.listing_idx = torch.arange(1, 1 + len(df))

        # labels on all nodes (only listing nodes matter for loss/metrics)
        g.y = torch.zeros(g.num_nodes)
        g.y[g.listing_idx] = torch.tensor(labels, dtype=torch.float)

        # -------- train/val node masks (optional) --------
        if add_masks:
            tr, vl = train_test_split(
                g.listing_idx.numpy(),
                test_size=0.1,
                stratify=labels,
                random_state=42,
            )
            g.train_mask = torch.zeros(g.num_nodes, dtype=torch.bool)
            g.val_mask = torch.zeros_like(g.train_mask)
            g.train_mask[tr] = True
            g.val_mask[vl] = True
        return g


# ────────────────────────────────────────────────────────────────
#                      GNN ENCODER (GAT)
# ────────────────────────────────────────────────────────────────
class GCN(torch.nn.Module):
    """
    Two-layer Graph-Attention network producing:

    * 256-dim hidden
    * 128-dim final embedding
    * 1-dim logit for binary desirability (via a third GAT layer)

    The **embed** method exposes the 128-dim vectors for stacking.
    """

    def __init__(self, in_dim: int = 8, drop: float = 0.4) -> None:
        super().__init__()
        self.g1 = GATConv(in_dim, 256)
        self.g2 = GATConv(256, 128)
        self.out = GATConv(128, 1)
        self.act = torch.nn.LeakyReLU()
        self.drop = torch.nn.Dropout(drop)

    # -------------- helpers --------------
    def embed(self, g: Data) -> torch.Tensor:
        h = self.drop(self.act(self.g1(g.x, g.edge_index)))
        h = self.drop(self.act(self.g2(h, g.edge_index)))
        return h

    # -------------- forward --------------
    def forward(self, g: Data) -> torch.Tensor:
        h = self.embed(g)
        return self.out(h, g.edge_index).squeeze()

    # -------------- persistence -----------
    def save(self, p: Path) -> None:
        torch.save(self.state_dict(), p)

    def load(self, p: Path) -> None:
        self.load_state_dict(torch.load(p, map_location="cpu"))


# ────────────────────────────────────────────────────────────────
#                  TABULAR STACKING HEAD
# ────────────────────────────────────────────────────────────────
class IntentHead:
    """
    Tabular-plus-embedding classifier with optional
    *mutual-information* feature pruning and probability calibration.

    Parameters
    ----------
    model_type : {'bayes', 'logreg'}
        Base learner family.
    calib : {'sigmoid', 'isotonic'}
        Calibration method passed to `CalibratedClassifierCV`.
    top_k : int
        If > 0, keep only the *k* tabular features with highest MI
        against the label.
    """

    def __init__(self, model_type: str = "bayes", calib: str = "sigmoid", top_k: int = 0):
        self.model_type = model_type
        self.calib_kind = calib
        self.top_k = top_k

        # schema bookkeeping
        self.tab_cols: List[str] = []            # all one-hot columns
        self.keep_idx: np.ndarray | None = None  # MI-based filter
        self.feat_names: List[str] = []          # human-readable names

        # choose base learner
        self.base = (
            GaussianNB()
            if model_type == "bayes"
            else LogisticRegression(
                max_iter=12_000,
                C=1.0,
                class_weight="balanced",
                solver="lbfgs",
            )
        )
        self.clf: CalibratedClassifierCV  # set after fit

    # ---------------- internal helpers ----------------
    def _one_hot(self, df: pd.DataFrame) -> np.ndarray:
        """
        One-hot encode all *non-date* columns.
        Column order is locked on the first call to maintain consistency
        between train/predict invocations.
        """
        df0 = df.drop(columns=["sold_date"])
        dummies = pd.get_dummies(df0, dummy_na=True)

        # lock schema on first call
        if not self.tab_cols:
            self.tab_cols = dummies.columns.tolist()
        dummies = dummies.reindex(columns=self.tab_cols, fill_value=0)

        X = dummies.values
        if self.top_k > 0 and self.keep_idx is not None:
            X = X[:, self.keep_idx]
        return X

    # ---------------- training ----------------
    def fit(self, emb: np.ndarray, df: pd.DataFrame, y: np.ndarray) -> None:
        """
        Train base classifier + calibrator on
        `X = [embeddings | one-hot tabular]`.
        """
        X_tab = self._one_hot(df)

        # optional MI feature selection
        if self.top_k > 0:
            mi = mutual_info_classif(X_tab, y, discrete_features=True, random_state=42)
            self.keep_idx = np.argsort(mi)[-self.top_k :]
            X_tab = X_tab[:, self.keep_idx]

        X = np.hstack([emb, X_tab])

        # split for calibration
        Xb, Xc, yb, yc = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        self.base.fit(Xb, yb)
        self.clf = CalibratedClassifierCV(estimator=self.base, cv="prefit", method=self.calib_kind)
        self.clf.fit(Xc, yc)

        # store feature names for explainability
        n_emb = emb.shape[1]
        emb_names = [f"emb_{i}" for i in range(n_emb)]
        tab_names = (
            np.array(self.tab_cols)[self.keep_idx].tolist()
            if self.top_k > 0 and self.keep_idx is not None
            else self.tab_cols
        )
        self.feat_names = emb_names + tab_names

    # ---------------- inference ----------------
    def predict_proba(self, emb: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        X_tab = self._one_hot(df)
        if self.top_k > 0 and self.keep_idx is not None:
            X_tab = X_tab[:, self.keep_idx]
        X = np.hstack([emb, X_tab])
        return self.clf.predict_proba(X)[:, 1]

    # ---------------- explainability ----------------
    def explain(
        self, emb_row: np.ndarray, df_row: pd.Series, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Return the *top-k* (feature, Δ log-odds) contributions for a
        single listing. Positive values push towards an offer.
        """
        # legacy checkpoints may not store feat_names
        if not getattr(self, "feat_names", []):
            n_emb = len(emb_row)
            emb_names = [f"emb_{i}" for i in range(n_emb)]
            tab_names = (
                np.array(self.tab_cols)[self.keep_idx].tolist()
                if self.top_k > 0 and self.keep_idx is not None
                else self.tab_cols
            )
            self.feat_names = emb_names + tab_names

        # rebuild full feature vector
        X_tab = self._one_hot(df_row.to_frame().T)
        if self.top_k > 0 and self.keep_idx is not None:
            X_tab = X_tab[:, self.keep_idx]
        x = np.hstack([emb_row.reshape(1, -1), X_tab]).flatten()

        # coefficient-based or NB-likelihood contribution
        if hasattr(self.base, "coef_"):  # LogisticRegression
            contrib = self.base.coef_.flatten() * x
        else:  # GaussianNB
            num = -0.5 * ((x - self.base.theta_[1]) ** 2) / self.base.sigma_[1]
            den = -0.5 * ((x - self.base.theta_[0]) ** 2) / self.base.sigma_[0]
            contrib = num - den

        feats = list(zip(self.feat_names, contrib))
        feats.sort(key=lambda t: abs(t[1]), reverse=True)
        return feats[:top_k]

    # ---------------- persistence ----------------
    def save(self, p: Path) -> None:
        joblib.dump(self, p)

    @staticmethod
    def load(p: Path) -> "IntentHead":
        obj: "IntentHead" = joblib.load(p)
        if not hasattr(obj, "feat_names"):
            obj.feat_names = []
        return obj


# ────────────────────────────────────────────────────────────────
#           SIMPLE TWO-PARAMETER THRESHOLD POLICY
# ────────────────────────────────────────────────────────────────
class StoppingPolicy:
    """`offer = (p ≥ τ) ∧ (1−p ≤ δ)`."""

    def __init__(self, tau: float = 0.7, delta: float = 0.10) -> None:
        self.tau = tau
        self.delta = delta

    def should_offer(self, p: float) -> bool:
        return p >= self.tau and (1.0 - p) <= self.delta


# ────────────────────────────────────────────────────────────────
#                 METRIC CONVENIENCE HELPERS
# ────────────────────────────────────────────────────────────────
class Metrics:
    """Tiny drop-in metrics helper (no sklearn dependency)."""

    # ------------- F-scores -------------
    @staticmethod
    def f1(y: np.ndarray, p: np.ndarray) -> float:
        tp = ((p == 1) & (y == 1)).sum()
        fp = ((p == 1) & (y == 0)).sum()
        fn = ((p == 0) & (y == 1)).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)

    # ------------- pretty print ---------
    @staticmethod
    def report(tag: str, y: np.ndarray, p: np.ndarray) -> None:
        tp = ((p == 1) & (y == 1)).sum()
        tn = ((p == 0) & (y == 0)).sum()
        fp = ((p == 1) & (y == 0)).sum()
        fn = ((p == 0) & (y == 1)).sum()
        acc = (tp + tn) / len(y)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = Metrics.f1(y, p)
        print(
            f"{tag} acc={acc:.3f}  prec={prec:.2f}  rec={rec:.2f}  f1={f1:.2f}"
        )
        print(f"confusion: TP {tp} | FP {fp} | FN {fn} | TN {tn}")


# ────────────────────────────────────────────────────────────────
#                   EARLY-STOP GNN TRAINER
# ────────────────────────────────────────────────────────────────
class GraphTrainer:
    """
    Minibatch neighbour-sampling trainer with early stopping on
    validation F1. Designed for *single graph* transductive training.
    """

    def __init__(
        self,
        g: Data,
        m: GCN,
        epochs: int = 60,
        patience: int = 10,
        lr: float = 1e-3,
        seed: int = 42,
    ):
        set_seed(seed)
        self.g, self.m, self.eps, self.pat = g, m, epochs, patience

        # balanced BCE loss
        pos = int(g.y[g.train_mask].sum())
        neg = int(g.train_mask.sum()) - pos
        self.crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1)]))

        self.opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)

        gen = torch.Generator().manual_seed(seed)
        self.loader = NeighborLoader(
            g,
            input_nodes=g.train_mask.nonzero(as_tuple=True)[0],
            num_neighbors=[15, 10],
            batch_size=256,
            shuffle=True,
            generator=gen,
            persistent_workers=False,
        )

        # helpful init for final bias
        with torch.no_grad():
            if hasattr(m, "out") and hasattr(m.out, "bias"):
                m.out.bias.zero_()

    # ---------------- training loop ----------------
    def run(self) -> None:
        best_f1, best_state = -1.0, None
        patience_left = self.pat

        for ep in range(1, self.eps + 1):
            # ----- train phase -----
            self.m.train()
            for batch in self.loader:
                self.opt.zero_grad()
                out = self.m(batch)
                loss = self.crit(out[batch.train_mask], batch.y[batch.train_mask])
                loss.backward()
                self.opt.step()

            # ----- validation phase -----
            self.m.eval()
            with torch.no_grad():
                logits = self.m(self.g)[self.g.val_mask]
                preds = (torch.sigmoid(logits) >= 0.5).int()
                f1 = Metrics.f1(self.g.y[self.g.val_mask].int(), preds)
            print(f"epoch {ep:03d}  val-F1={f1:.3f}")

            # ----- early stopping -----
            if f1 > best_f1 + 1e-4:
                best_f1 = f1
                patience_left = self.pat
                best_state = copy.deepcopy(self.m.state_dict())
            else:
                patience_left -= 1
            if patience_left == 0:
                print("early-stop triggered")
                break

        # restore best weights
        if best_state is not None:
            self.m.load_state_dict(best_state)

        # final report on *train* split
        self.m.eval()
        with torch.no_grad():
            logits = self.m(self.g)[self.g.train_mask]
            p = (torch.sigmoid(logits) >= 0.5).int()
        Metrics.report("train", self.g.y[self.g.train_mask].int(), p)


# ────────────────────────────────────────────────────────────────
#               LIGHTWEIGHT HELPERS FOR CLI TASKS
# ────────────────────────────────────────────────────────────────
def predict_probs_graph(g: Data, mdl: Path) -> np.ndarray:
    """Utility: load GNN model & output *sigmoid* probabilities."""
    m = GCN()
    m.load(mdl)
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(g)).numpy()


def sweep_graph(g: Data, mdl: Path, start=0.3, end=0.9, step=0.05) -> None:
    """
    Print precision/recall/F1 across a range of thresholds on a *graph*-only
    model.  Labels must be present on `g`.
    """
    prob = predict_probs_graph(g, mdl)[g.listing_idx]
    y = g.y[g.listing_idx].int().numpy() if hasattr(g, "y") else None

    print("thr  prec  rec  f1")
    t = start
    while t <= end + 1e-9:
        p = (prob >= t).astype(int)
        if y is None:
            print(f"{t:.2f}  --   --   --")
        else:
            tp = ((p == 1) & (y == 1)).sum()
            fp = ((p == 1) & (y == 0)).sum()
            fn = ((p == 0) & (y == 1)).sum()
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = Metrics.f1(y, p)
            print(f"{t:.2f}  {prec:.2f}  {rec:.2f}  {f1:.2f}")
        t += step


# ────────────────────────────────────────────────────────────────
#                     OPTION-VALUE PLANNER
# ────────────────────────────────────────────────────────────────
class OptionValuePlanner:
    """
    Very light Bayesian stop-or-wait module.

    Assumptions
    -----------
    * The next week's *k* listings are IID draws from the empirical
      posterior distribution seen so far *after* sold-filtering.
    * Regret if a strictly better listing appears is linear
      (`regret_cost`).
    * Waiting incurs a fixed utility loss (`wait_cost`).

    Decision Rule
    -------------
        P(better) · regret_cost   ≤   wait_cost

    where ``P(better) = 1 − F(p_best) ** k`` and *F* is the empirical CDF.
    """

    def __init__(
        self,
        wait_cost: float = 0.02,
        regret_cost: float = 0.10,
        k_estimate: int = 30,
    ):
        self.wait_cost = float(wait_cost)
        self.regret_cost = float(regret_cost)
        self.k = max(1, int(k_estimate))
        self._scores: np.ndarray | None = None

    # ---------------- API ----------------
    def fit(self, scores: np.ndarray) -> "OptionValuePlanner":
        """Store full history of posterior samples."""
        self._scores = np.asarray(scores, dtype=float)
        return self

    def prob_better(self, p_best: float) -> float:
        """Probability that *at least one* better listing appears next week."""
        F = self._cdf(p_best)
        return 1.0 - (F ** self.k)

    def should_stop(self, p_best: float) -> Tuple[bool, float, float]:
        """
        Returns
        -------
        stop     : bool
        p_better : float
            Probability a better listing appears next week.
        delta    : float
            Expected-regret-difference  (ER_stop − ER_wait).
        """
        pb = self.prob_better(p_best)
        er_stop = pb * self.regret_cost
        er_wait = self.wait_cost
        return er_stop <= er_wait, pb, er_stop - er_wait

    # ---------------- internals ----------------
    def _cdf(self, x: float) -> float:
        if self._scores is None or len(self._scores) == 0:
            return 0.0
        return float((self._scores <= x).mean())


def _update_hist_scores(hist_path: Path, new_scores: np.ndarray) -> np.ndarray:
    """
    Append new posterior samples to *hist_path* (``.npy``) and return the
    full concatenated history.
    """
    if hist_path.exists():
        hist_scores = np.load(hist_path)
        hist_scores = np.concatenate([hist_scores, new_scores])
    else:
        hist_scores = new_scores
    np.save(hist_path, hist_scores)
    return hist_scores


# ────────────────────────────────────────────────────────────────
#            STACK-LEVEL THRESHOLD SWEEP (IntentHead)
# ────────────────────────────────────────────────────────────────
def sweep_stack(
    csv: Path,
    gnn_pt: Path,
    head_pkl: Path,
    start: float,
    end: float,
    step: float,
    metric: str = "f1",
    beta: float = 1.0,
) -> None:
    """
    Brute-force sweep over τ for the *stacked* model \
    (GNN + tabular head).

    ``metric == "fbeta"`` uses user-supplied β.
    """

    def f_beta(p: float, r: float, b: float) -> float:
        return 0.0 if p == 0 and r == 0 else (1 + b * b) * p * r / (b * b * p + r)

    # ---------- load data & models ----------
    df = pd.read_csv(csv)
    g = ListingGraphTG().from_df(df, add_masks=False)
    m = GCN()
    m.load(gnn_pt)
    m.eval()
    with torch.no_grad():
        emb = m.embed(g)[g.listing_idx].numpy()
    head = IntentHead.load(head_pkl)
    prob = head.predict_proba(emb, df)

    # zero-out listings already sold
    prob[df["sold_date"].notna().values] = 0.0
    y = g.y[g.listing_idx].int().numpy()

    # ---------- sweep ----------
    best_val, best_thr = -1.0, None
    print("thr  prec  rec  score")
    t = start
    while t <= end + 1e-9:
        p = (prob >= t).astype(int)
        tp = ((p == 1) & (y == 1)).sum()
        fp = ((p == 1) & (y == 0)).sum()
        fn = ((p == 0) & (y == 1)).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        score = Metrics.f1(y, p) if metric == "f1" else f_beta(prec, rec, beta)
        print(f"{t:.02f}  {prec:.02f}  {rec:.02f}  {score:.02f}")
        if score > best_val:
            best_val, best_thr = score, t
        t += step

    print(f"\nBEST τ={best_thr:.02f}  {metric.upper()}={best_val:.02f}")


# ────────────────────────────────────────────────────────────────
#                      CLI ARGUMENT PARSER
# ────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description="Graph-Stop Home Hunter CLI")
# graph-level jobs
ap.add_argument("--save-graph", nargs=2)
ap.add_argument("--train-graph", nargs=2)
ap.add_argument("--predict-graph", nargs=2)
ap.add_argument("--sweep-graph", nargs=2)
# stack jobs
ap.add_argument("--train-stack", nargs=3)
ap.add_argument("--predict-stack", nargs=3)
ap.add_argument("--sweep-stack", nargs=3)
# simulations
ap.add_argument("--simulate", nargs="+")
ap.add_argument("--simulate-pretrained", nargs="+")
# experiment flags
ap.add_argument("--stack-model", choices=["bayes", "logreg"], default="bayes")
ap.add_argument("--top-k-tab", type=int, default=0)
ap.add_argument("--calib", choices=["sigmoid", "isotonic"], default="sigmoid")
ap.add_argument("--metric", choices=["f1", "fbeta"], default="f1")
ap.add_argument("--beta", type=float, default=1.0)
# generic params
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--lr", type=float, default=1e-2)
ap.add_argument("--thr", type=float, default=0.50)
ap.add_argument("--delta", type=float, default=0.10)
ap.add_argument("--start", type=float, default=0.30)
ap.add_argument("--end", type=float, default=0.90)
ap.add_argument("--step", type=float, default=0.05)
# OVP
ap.add_argument("--wait-cost", type=float, default=0.02, help="Utility loss for waiting one week")
ap.add_argument("--regret-cost", type=float, default=0.10, help="Regret if a better listing appears")
ap.add_argument("--k-next-week", type=int, default=30, help="Expected new listings next week")
ap.add_argument("--quiet-metrics", action="store_true", help="Suppress per-call metric printing (used by simulate_weeks)")
ap.add_argument(
    "--explain-id",
    type=int,
    metavar="N",
    help="Explain features for listing *N* (1-based) after predict-stack",
)
args, _ = ap.parse_known_args()

# ────────────────────────────────────────────────────────────────
#                 CLI SUB-COMMAND IMPLEMENTATIONS
# ────────────────────────────────────────────────────────────────

# ───────────── graph-only sub-commands ─────────────
if args.save_graph:
    # Save the graph representation of a weekly CSV for later reuse
    csv, out = map(Path, args.save_graph)
    g = ListingGraphTG().from_csv(csv)
    torch.save(g, out)
    print("graph saved →", out.resolve())
    sys.exit()

if args.train_graph:
    # Train the GNN encoder (no tabular head) on a graph
    gp, mdl = map(Path, args.train_graph)
    g = _safe_torch_load(gp) if gp.suffix == ".pt" else ListingGraphTG().from_csv(gp)
    m = GCN()
    GraphTrainer(g, m, args.epochs, args.patience, args.lr).run()
    m.save(mdl)
    print("model→", mdl.resolve())
    sys.exit()

if args.predict_graph:
    # Pure-GNN inference; optional metrics if labels exist
    src, mdl = map(Path, args.predict_graph)
    g = _safe_torch_load(src) if src.suffix == ".pt" else ListingGraphTG().from_csv(
        src, add_masks=False
    )
    prob = predict_probs_graph(g, mdl)
    pred = (prob >= args.thr).astype(int)
    if hasattr(g, "y"):
        Metrics.report("test", g.y[g.listing_idx].int().numpy(), pred[g.listing_idx])
    else:
        print("predictions ready (no labels)")
    sys.exit()

if args.sweep_graph:
    # Precision/recall/F1 sweep for a graph-only model
    gpt, mdl = map(Path, args.sweep_graph)
    g = _safe_torch_load(gpt) if gpt.suffix == ".pt" else ListingGraphTG().from_csv(
        gpt, add_masks=False
    )
    sweep_graph(g, mdl, args.start, args.end, args.step)
    sys.exit()


# ───────────── stack sub-commands (GNN + head) ─────────────
if args.train_stack:
    # 1) GNN embeddings → 2) IntentHead training → 3) save .pkl
    csv, gnn_pt, head_pkl = map(Path, args.train_stack)
    g = ListingGraphTG().from_csv(csv)
    m = GCN()
    m.load(gnn_pt)
    m.eval()
    with torch.no_grad():
        emb = m.embed(g)[g.listing_idx].numpy()
    df = pd.read_csv(csv)
    y = g.y[g.listing_idx].int().numpy()

    head = IntentHead(
        model_type=args.stack_model, calib=args.calib, top_k=args.top_k_tab
    )
    head.fit(emb, df, y)
    head.save(head_pkl)
    print("stack trained →", head_pkl.resolve())
    sys.exit()


# ───────────── predict-stack — single week CSV ─────────────
if args.predict_stack:
    csv, gnn_pt, head_pkl = map(Path, args.predict_stack)

    # ❶ Load raw CSV (tabular features + sold flag)
    df = pd.read_csv(csv)

    # ❷ Build PyG graph (no masks needed for inference)
    g = ListingGraphTG().from_df(df, add_masks=False)

    # ❸ GNN → embeddings
    m = GCN()
    m.load(gnn_pt)
    m.eval()
    with torch.no_grad():
        emb = m.embed(g)[g.listing_idx].numpy()

    # ❹ Tabular head → posterior probability per listing
    head = IntentHead.load(head_pkl)
    prob = head.predict_proba(emb, df)

    # ❺ Zero-out listings already sold
    sold_mask = df["sold_date"].notna().values
    prob[sold_mask] = 0.0

    # ❻ Offer threshold τ
    pred = (prob >= args.thr).astype(int)

    # ❼ Metrics (optional)
    if not args.quiet_metrics:
        print("stack metrics:")
        if hasattr(g, "y"):
            Metrics.report("stack", g.y[g.listing_idx].int().numpy(), pred)

    # ❽ Legacy 2-parameter heuristic
    pol = StoppingPolicy(tau=args.thr, delta=args.delta)
    #for i, pst in enumerate(prob[:5]):
    #    print(f"id {i+1:3d}  post={pst:.02f}  offer? {pol.should_offer(pst)}")

    # ❾ ------------- OPTION-VALUE PLANNER -------------
    hist_path = head_pkl.with_suffix(".scores.npy")
    hist_scores = _update_hist_scores(hist_path, prob)

    ovp = OptionValuePlanner(
        wait_cost=args.wait_cost,
        regret_cost=args.regret_cost,
        k_estimate=args.k_next_week,
    ).fit(hist_scores)

    p_best = prob.max()
    stop, p_better, delta = ovp.should_stop(p_best)

    print(
        "\nOVP decision:  " + ("💰  **PLACE OFFER NOW**" if stop else "⏳  WAIT")
    )
    print(f"  • current best posterior  : {p_best:.3f}")
    print(f"  • P(better next week)     : {p_better:.2%}")
    print(f"  • Δ(Expected regret)      : {delta:+.4f}")

    # ――― Optional explain-one-listing ―――
    if args.explain_id is not None:
        idx = args.explain_id - 1
        feats = head.explain(emb[idx], df.iloc[idx])
        print(f"\nTop feature contributions for listing {args.explain_id}:")
        for name, val in feats:
            print(f"  {name:<25s} {val:+.3f}")
    sys.exit()


# ───────────── sweep-stack ─────────────
if args.sweep_stack:
    csv, gnn_pt, head_pkl = map(Path, args.sweep_stack)
    sweep_stack(
        csv,
        gnn_pt,
        head_pkl,
        args.start,
        args.end,
        args.step,
        metric=args.metric,
        beta=args.beta,
    )
    sys.exit()


# ───────────── simulate-pretrained (batch weeks) ─────────────
if args.simulate_pretrained:
    # Usage: --simulate-pretrained gnn.pt head.pkl week0.csv week1.csv ...
    gnn_pt, head_pkl, *weeks = args.simulate_pretrained
    m = GCN()
    m.load(Path(gnn_pt))
    m.eval()
    head = IntentHead.load(Path(head_pkl))

    hist_path = Path(head_pkl).with_suffix(".scores.npy")
    if hist_path.exists():
        hist_path.unlink()  # reset history between runs

    for idx, csv_path in enumerate(weeks):
        df = pd.read_csv(csv_path)
        # remove sold listings for live inference
        df = df[df["sold_date"].isna()].reset_index(drop=True)
        g = ListingGraphTG().from_df(df, add_masks=False)

        with torch.no_grad():
            emb = m.embed(g)[g.listing_idx].numpy()
            post = head.predict_proba(emb, df)

        # accumulate history & run OVP
        hist_scores = _update_hist_scores(hist_path, post)
        ovp = OptionValuePlanner(
            wait_cost=args.wait_cost,
            regret_cost=args.regret_cost,
            k_estimate=len(df),
        ).fit(hist_scores)

        p_best = post.max()
        stop, pb, _ = ovp.should_stop(p_best)

        print(f"\n════════ Week {idx} ({csv_path}) ════════")
        offers = np.where(post >= args.thr)[0] + 1
        print(f"Offer IDs (basic τ)  : {offers.tolist()}")
        print("OVP decision        :", "OFFER NOW" if stop else "WAIT", f"(P_better={pb:.1%})")

        if hasattr(g, "y"):
            Metrics.report(
                "week", g.y[g.listing_idx].int().numpy(), (post >= args.thr).astype(int)
            )
    sys.exit()


# ───────────── simulate (rolling retrain) ─────────────
if args.simulate:
    print("Rolling simulation not included in this excerpt.")
    sys.exit()


# ────────────────────────────────────────────────────────────────
#                             DEFAULT
# ────────────────────────────────────────────────────────────────
print("Nothing to do – run with --help")
