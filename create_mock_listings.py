#!/usr/bin/env python
"""
data_generator.py  –  “Graph-Stop” weekly snapshot generator
──────────────────────────────────────────────────────────────────
Creates **N_WEEKS** CSV files (`week_0.csv` … `week_9.csv`) under
`data_snapshots/`.  Each file is a full listing table for that week
including:

* a core of *active* listings carried forward from previous weeks
* `N_WEEKLY_SOLD` randomly selected listings that become *sold* this week
  (column `sold_date` stamped with the Monday of that week)
* `N_WEEKLY_NEW` freshly generated listings
* PLUS one handcrafted **“golden” listing** in **week 6** to ensure the
  model sees an obvious best-in-class candidate (useful for demos).

Attributes that matter to the downstream model—price, beds, size,
travel-time, boolean prefs, livability—are sampled with realistic
correlations and noise.  Livability is *fixed per suburb* so the GNN
can learn a meaningful geographic signal.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# ═════════════════════════════ CONFIG ═════════════════════════════
N_INITIAL_LISTINGS = 100
N_WEEKLY_SOLD = 10
N_WEEKLY_NEW = 10
N_WEEKS = 10                          # week_0 … week_9
OUT_DIR = Path("data_snapshots")
RNG = np.random.default_rng(42)

# (suburb, distance-to-CBD [km])
SUBURBS: list[Tuple[str, int]] = [
    ("Melbourne CBD", 0),
    ("Carlton", 2),
    ("Richmond", 3),
    ("Brunswick", 6),
    ("Footscray", 6),
    ("Balwyn", 12),
    ("Box Hill", 15),
    ("Werribee", 32),
    ("Sunbury", 40),
    ("Geelong", 65),
    ("St Kilda", 5),
]
# Deterministic livability per suburb (0-25 scale)
LIVABILITY = {
    "Melbourne CBD": 15,
    "Carlton": 18,
    "Richmond": 16,
    "Brunswick": 12,
    "Footscray": 10,
    "Balwyn": 22,
    "Box Hill": 14,
    "Werribee": 8,
    "Sunbury": 9,
    "Geelong": 11,
    "St Kilda": 20,
}
BEDS_OPTS = [1, 2, 3, 4]

# ═══════════════════ LISTING-LEVEL HELPERS ════════════════════════
def _random_suburb() -> Tuple[str, int]:
    """Return (suburb_name, km_to_CBD) chosen uniformly."""
    idx = RNG.integers(len(SUBURBS))
    return SUBURBS[idx]


def _make_listing(listing_id: int) -> dict:
    """Generate *one* realistically noisy listing."""
    suburb, km = _random_suburb()
    beds = int(RNG.choice(BEDS_OPTS, p=[0.20, 0.45, 0.25, 0.10]))

    # ---------- numeric features ----------
    price = 450_000 + beds * 110_000 + RNG.normal(0, 25_000) - km * 4_000
    price = int(round(price, -3))  # round to nearest $1 000
    internal_m2 = round(np.clip(RNG.normal(55 + beds * 20, 8), 35, None), 1)
    balcony_m2 = round(abs(RNG.normal(8 if beds <= 2 else 12, 5)), 1)
    travel_time = int(np.clip(RNG.normal(12 + km * 0.9, 8), 5, 90))

    # ---------- boolean / categorical ----------
    return dict(
        id=listing_id,
        asking_price=price,
        body_corp=int(max(0, RNG.normal(2_200, 600))),
        internal_m2=internal_m2,
        balcony_m2=balcony_m2,
        beds=beds,
        north_facing=RNG.random() < 0.35,
        suburb=suburb,
        has_parking=RNG.random() < 0.55,
        has_storage=RNG.random() < 0.40,
        has_solar=RNG.random() < 0.15,
        near_shops=RNG.random() < 0.70,
        livability=LIVABILITY[suburb],
        condition=RNG.choice(["new", "good", "needs TLC"], p=[0.25, 0.55, 0.20]),
        has_cladding=RNG.random() < 0.05,
        outdoor_space=RNG.random() < 0.50,
        sold_date=pd.NaT,  # not sold yet
        travel_time_mins=travel_time,
    )


# ---------------------------------------------------------------- #
# A deliberately over-the-top listing to guarantee a posterior peak
# ---------------------------------------------------------------- #
def _golden_listing(listing_id: int) -> dict:
    """
    Handcraft a “perfect” property that smashes every preference weight
    (cheap, huge, close, solar, parking, etc.). Appears only in Week 6.
    """
    suburb = "St Kilda"  # in preferred_areas + high livability (20)
    return dict(
        id=listing_id,
        asking_price=320_000,
        body_corp=400,
        internal_m2=150.0,
        balcony_m2=20.0,
        beds=4,
        north_facing=True,
        suburb=suburb,
        has_parking=True,
        has_storage=True,
        has_solar=True,
        near_shops=True,
        livability=LIVABILITY[suburb],
        condition="new",
        has_cladding=False,
        outdoor_space=True,
        sold_date=pd.NaT,
        travel_time_mins=6,
    )


# ═════════════════ WEEKLY STATE-UPDATE UTILITIES ═════════════════
def _mark_random_sold(df: pd.DataFrame, n_sold: int, week_start: dt.date) -> pd.DataFrame:
    """Randomly choose *n_sold* active listings and stamp sold_date."""
    active_idx = df[df["sold_date"].isna()].index
    to_sell = RNG.choice(active_idx, size=min(n_sold, len(active_idx)), replace=False)
    df.loc[to_sell, "sold_date"] = pd.Timestamp(week_start)
    return df


def _append_new(df: pd.DataFrame, n_new: int, *, inject_golden: bool = False) -> pd.DataFrame:
    """Add either n_new random listings (plus optional golden one)."""
    next_id = df["id"].max() + 1
    new_rows = [_make_listing(next_id + i) for i in range(n_new)]
    if inject_golden:
        new_rows.append(_golden_listing(next_id + n_new))
    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


# ═════════════════════ CSV WRITER (no CRLF) ═══════════════════════
def _write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write CSV with Unix line endings regardless of platform."""
    with path.open("w", newline="", encoding="utf-8") as f:
        df.to_csv(f, index=False, lineterminator="\n")


# ═════════════════════════════ MAIN ══════════════════════════════
def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    # ------- week 0 -------
    df = pd.DataFrame(_make_listing(i + 1) for i in range(N_INITIAL_LISTINGS))
    _write_csv(df, OUT_DIR / "week_0.csv")
    print("✓ week_0.csv")

    # Monday of the current week (for deterministic sold_date stamps)
    monday = dt.date.today() - dt.timedelta(days=dt.date.today().weekday())

    # ------- weeks 1 … N_WEEKS-1 -------
    for wk in range(1, N_WEEKS):
        df = _mark_random_sold(df, N_WEEKLY_SOLD, monday + dt.timedelta(weeks=wk))
        df = _append_new(df, N_WEEKLY_NEW, inject_golden=(wk == 6))
        _write_csv(df, OUT_DIR / f"week_{wk}.csv")
        print(f"✓ week_{wk}.csv {'(golden)' if wk == 6 else ''}")

    print("All snapshots saved →", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
