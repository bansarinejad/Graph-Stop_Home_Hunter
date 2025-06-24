#!/usr/bin/env python
"""
simulate_weeks.py
─────────────────
Replay a *chronological* list of weekly CSV snapshots through
`graph_home_hunter.py --predict-stack …`, printing the Option-Value
Planner (OVP) verdict for every week.

Why a wrapper?
--------------
The main CLI runs a *single* week at a time.  This helper automates the
week-after-week shell-out so you can watch the OVP’s stop/-wait logic
unfold without writing a loop by hand.

Example
-------
python simulate_weeks.py week5.csv week6.csv week7.csv \
    --gnn gnn_v1.pt --head logreg_head.pkl --thr 0.95
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ────────────────────────── CLI ARGUMENTS ──────────────────────────
parser = argparse.ArgumentParser(description="Batch-simulate consecutive weeks.")
parser.add_argument("csv", nargs="+", help="Weekly CSV files in chronological order")
parser.add_argument("--gnn", required=True, help="Pre-trained GNN .pt file")
parser.add_argument("--head", required=True, help="Pre-trained IntentHead .pkl file")
parser.add_argument("--thr", type=float, default=0.95, help="Offer threshold τ")
parser.add_argument("--wait-cost", type=float, default=0.02, help="Weekly wait-cost")
parser.add_argument("--regret-cost", type=float, default=2.0, help="Regret-cost")
parser.add_argument("--k-next-week", type=int, default=10,
                    help="Expected # new listings next week "
                         "(0 ⇒ len(current CSV))")
parser.add_argument(
    "--script",
    default="graph_home_hunter.py",
    help="Main CLI script to call (default: graph_home_hunter.py)",
)
args = parser.parse_args()

# Make sure CSVs are processed oldest → newest even if user types them out of
# order. 
args.csv.sort(key=lambda p: Path(p).name)

# ──────────────────────────── MAIN LOOP ───────────────────────────
for week_idx, csv_path in enumerate(args.csv, start=0):
    # If the user gave --k-next-week 0, derive k from current file size.
    k_next = args.k_next_week or len(pd.read_csv(csv_path))

    # Compose the command line for one week
    cmd: list[str] = [
        sys.executable,
        args.script,
        "--predict-stack",
        csv_path,
        args.gnn,
        args.head,
        "--thr",          str(args.thr),
        "--wait-cost",    str(args.wait_cost),
        "--regret-cost",  str(args.regret_cost),
        "--k-next-week",  str(k_next),
        "--quiet-metrics",                 # suppress per-week F1 spam
    ]

    # Pretty-print then execute
    print(f"\n🏠  Week {week_idx}: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)
