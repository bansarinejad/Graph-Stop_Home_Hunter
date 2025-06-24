#!/usr/bin/env python
"""
make_training_data.py  –  noisy listings, fixed livability
───────────────────────────────────
Synthetic **listing generator** that outputs two CSV files:

* **train_listings.csv** – 80 % of rows
* **test_listings.csv**  – 20 % of rows

Each row represents a Melbourne-area apartment with noisy numeric /
boolean features and a binary **label** that approximates “worth making
an offer” for the top 5% of the listings that are closest matches to the 
user criteria. Labels are produced by the same *preference rules* used in
the online demo so the GNN + tabular stack can be trained *supervised*.

Usage
-----
python make_training_data_v4_fixed_live.py
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ══════════════════════════════ GLOBAL KNOBS ═══════════════════════════════
N_TOTAL = 20_000        # rows to generate
TOP_PERCENT = 0.05      # label = 1 for top 5 % of preference-scores

# Gaussian noise std-dev on numeric columns
SIGMA_PRICE = 40_000
SIGMA_INT_M2 = 10
SIGMA_BAL_M2 = 5
SIGMA_BODY_CORP = 800
SIGMA_TRAVEL = 12

MISS_RATE_BOOL = 0.08   # % chance to wipe a boolean → NaN
LABEL_FLIP = 0.05       # % of labels to invert (real-world noise)

RNG = np.random.default_rng(0)  # reproducible randomness
OUT_DIR = Path(".")

# Suburb tuples:  (name, km_to_CBD, **FIXED** livability score 0-25)
SUBURBS = [
    ("Melbourne CBD", 0, 15),
    ("Carlton", 2, 18),
    ("Richmond", 3, 16),
    ("Brunswick", 6, 12),
    ("Footscray", 6, 10),
    ("Balwyn", 12, 22),
    ("Box Hill", 15, 14),
    ("Werribee", 32, 8),
    ("Sunbury", 40, 9),
    ("Geelong", 65, 11),
    ("St Kilda", 5, 20),
]

BEDS_OPTS = [1, 2, 3, 4]
COND_OPTS = ["new", "good", "needs TLC"]

# ═════════════════════ USER-PREFERENCE SCORE (unchanged) ═══════════════════
@dataclass
class UserPreferences:
    max_price: float = 600_000
    max_body_corp: float = 6_000
    min_beds: int = 2
    min_internal_m2: float = 50
    good_condition: bool = True
    no_cladding: bool = True
    max_travel_mins: int = 45
    near_shops: bool = True
    min_livability: float = 0.0

    # Optional boolean prefs (None = “don’t care”)
    has_parking: Optional[bool] = None
    has_storage: Optional[bool] = None
    has_solar: Optional[bool] = None
    north_facing: Optional[bool] = None
    outdoor_space: Optional[bool] = None

    preferred_areas: List[str] = field(
        default_factory=lambda: [
            "Melbourne CBD",
            "Brunswick",
            "St Kilda",
            "Carlton",
            "Hawthorn",
        ]
    )

    # Weights for soft preferences
    w_top: float = 1.0
    w_lower: float = 0.6
    w_lowest: float = 0.3


class PreferenceScorer:
    """Rule-based soft scorer used to derive binary labels."""

    def __init__(self, p: UserPreferences):
        self.p = p

    # -------------- helpers --------------
    @staticmethod
    def _b(v: Any) -> Optional[bool]:
        """Convert common truthy / falsy strings to Python bool."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            m = {
                "yes": True,
                "true": True,
                "y": True,
                "1": True,
                "no": False,
                "false": False,
                "n": False,
                "0": False,
            }
            return m.get(v.strip().lower())

    def _match(self, val: Any, pref: Optional[bool]) -> bool:
        """Return *True* if `pref` is None **or** values match."""
        return True if pref is None else self._b(val) == pref

    # -------------- public --------------
    def score(self, r: Dict[str, Any]) -> float:
        """
        Return a positive soft-score if the listing passes **all hard
        constraints**; otherwise -inf.
        """
        p = self.p
        hard_ok = (
            r["asking_price"] <= p.max_price
            and r["body_corp"] <= p.max_body_corp
            and r["beds"] >= p.min_beds
            and r["internal_m2"] >= p.min_internal_m2
            and (not p.good_condition or r["condition"] in ("good", "new"))
            and (not p.no_cladding or not r["has_cladding"])
            and r["travel_time_mins"] <= p.max_travel_mins
            and (not p.near_shops or r["near_shops"])
            and r["livability"] >= p.min_livability
        )
        if not hard_ok:
            return -math.inf

        # accumulate soft bonuses
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


# ═════════════════════════ LISTING GENERATOR ═══════════════════════
BOOL_COLS = [
    "has_parking",
    "has_storage",
    "has_solar",
    "north_facing",
    "near_shops",
    "outdoor_space",
]


def one_listing(idx: int) -> Dict[str, Any]:
    """
    Create **one** synthetic listing with realistic correlations
    (price↑ with beds, price↓ with distance, etc.).
    """
    # pick suburb & static attributes
    sb_idx = RNG.integers(len(SUBURBS))
    suburb, km, live_fixed = SUBURBS[sb_idx]
    beds = int(RNG.choice(BEDS_OPTS, p=[0.2, 0.45, 0.25, 0.10]))

    # numeric features with noise
    price = 450_000 + beds * 110_000 + RNG.normal(0, SIGMA_PRICE) - km * 4_000
    price = int(round(price, -3))  # nearest $1 000
    internal = max(32, int(RNG.normal(55 + beds * 20, SIGMA_INT_M2)))
    balcony = max(0, round(abs(RNG.normal(8 if beds <= 2 else 12, SIGMA_BAL_M2)), 1))
    travel = int(np.clip(RNG.normal(12 + km * 0.9, SIGMA_TRAVEL), 5, 90))

    # boolean columns (+ missingness)
    bool_vals = {
        "has_parking": RNG.random() < 0.55,
        "has_storage": RNG.random() < 0.40,
        "has_solar": RNG.random() < 0.15,
        "north_facing": RNG.random() < 0.35,
        "near_shops": RNG.random() < 0.70,
        "outdoor_space": RNG.random() < 0.50,
    }
    for k in BOOL_COLS:
        if RNG.random() < MISS_RATE_BOOL:
            bool_vals[k] = np.nan  # simulate missing data

    return dict(
        id=idx,
        asking_price=price,
        body_corp=int(max(0, RNG.normal(2_200, SIGMA_BODY_CORP))),
        internal_m2=internal,
        balcony_m2=balcony,
        beds=beds,
        suburb=suburb,
        livability=live_fixed,  # ★ fixed per suburb ★
        condition=RNG.choice(COND_OPTS, p=[0.25, 0.55, 0.20]),
        has_cladding=RNG.random() < 0.05,
        travel_time_mins=travel,
        sold_date=np.nan,  # not sold yet
        **bool_vals,
    )


# ═══════════════════ BUILD DATAFRAME & LABELS ═════════════════════
def build_dataset(n: int, top_pct: float) -> pd.DataFrame:
    """Generate *n* listings then assign noisy binary labels."""
    rows = [one_listing(i + 1) for i in range(n)]
    df = pd.DataFrame(rows)

    # soft scores → binary labels
    scorer = PreferenceScorer(UserPreferences())
    df["score"] = df.apply(lambda r: scorer.score(r.to_dict()), axis=1)

    valid = df["score"] > -math.inf
    cutoff = np.quantile(df.loc[valid, "score"], 1 - top_pct)
    df["label"] = ((df["score"] >= cutoff) & valid).astype(int)

    # label-flip noise
    flip_idx = RNG.choice(df.index, size=int(LABEL_FLIP * len(df)), replace=False)
    df.loc[flip_idx, "label"] ^= 1

    return df.drop(columns="score")


# ══════════════════════════ MAIN ENTRY ═══════════════════════════
def main() -> None:
    """Generate CSVs under current directory."""
    df = build_dataset(N_TOTAL, TOP_PERCENT).sample(frac=1, random_state=42)
    split = int(0.8 * len(df))

    OUT_DIR.mkdir(exist_ok=True)
    df.iloc[:split].to_csv(OUT_DIR / "train_listings.csv", index=False)
    df.iloc[split:].to_csv(OUT_DIR / "test_listings.csv", index=False)

    print(f"✓ train_listings.csv {split} rows")
    print(f"✓ test_listings.csv  {len(df) - split} rows")


if __name__ == "__main__":
    main()






