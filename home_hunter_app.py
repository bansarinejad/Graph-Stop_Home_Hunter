#!/usr/bin/env python
"""
home_hunter_app.py — Streamlit UI for the “Graph-Stop Home-Hunter” demo.

Version: v0.9.1 → *doc-clarified*

This script lets you replay a home-buying simulation week-by-week:

1. **Upload models** – a trained GNN encoder (`.pt`) and stacked tabular
   head (`.pkl`).
2. **Upload CSV snapshots** – one file per calendar week of listings.
3. **Tune decision parameters**  
   • `τ` – posterior threshold for “good enough” listings  
   • `wait_cost` – utility lost each week you keep searching  
   • `regret_cost` – pain of missing a better future listing  
   • `k_next` – expected number of *new* listings next week  
4. The UI runs an **Option-Value Planner (OVP)** that weighs expected
   regret against waiting and shows a banner:

      🟢 **RECOMMEND OFFER NOW**  
      🟡 **WAIT – better listing likely next week**

5. Each row is clickable for an explanatory breakdown of the model’s
   top-k feature contributions.

The script is intentionally **side-effect-free** except for Streamlit
state and temporary model files. It may be executed either via
`streamlit run home_hunter_app.py` or imported as a module.
"""
from __future__ import annotations

# ────────────────────── Standard library ──────────────────────
from pathlib import Path
from typing import Tuple, Set, List

# ──────────────────────── Third-party ─────────────────────────
import numpy as np
import pandas as pd
import streamlit as st
import torch

# ─────────────────────── Application ──────────────────────────
# Adjust the import path if you renamed `graph_home_hunter.py`
from graph_home_hunter import (
    ListingGraphTG,
    GCN,
    IntentHead,
    OptionValuePlanner,
)

# ╭──────────────────────────── UI SET-UP ─────────────────────────────╮
st.set_page_config(
    page_title="Graph-Stop Home-Hunter", page_icon="🏠", layout="wide"
)
st.title("🏠 Graph-Stop Home-Hunter")
# ╰────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────── SIDEBAR CONFIG ──────────────────────────╮
with st.sidebar:
    # ――― Model upload ―――
    st.header("Models")
    gnn_file = st.file_uploader("GNN model (.pt)", type="pt")
    head_file = st.file_uploader("Head model (.pkl)", type="pkl")

    # ――― Listing snapshots ―――
    st.divider()
    st.header("CSV snapshots")
    csv_files = st.file_uploader(
        "Weekly files (week0 → weekN)", type="csv", accept_multiple_files=True
    )

    # ――― Decision parameters ―――
    st.divider()
    st.header("Decision parameters")
    tau = st.slider("Offer threshold τ", 0.0, 1.0, 0.95, 0.01)
    wait_cost = st.slider("Wait-cost (weekly)", 0.0, 0.10, 0.02, 0.005)
    regret_cost = st.slider("Regret-cost", 0.0, 0.50, 0.35, 0.01)

    fixed_k = st.number_input("Fixed k (expected new listings)", 1, 500, 10, 1)
    auto_k = st.toggle("📈 Auto-detect k from arrivals", value=False)
    st.caption(
        "If auto-detect is ON, k = #listings that *did not* appear last "
        "week; otherwise the fixed value above is used."
    )
# ╰────────────────────────────────────────────────────────────────────╯


# ╭─────────────────────────── HELPERS ────────────────────────────────╮
@st.cache_resource(show_spinner="Loading models …")
def load_models(
    gnn_bytes: bytes, head_bytes: bytes
) -> Tuple[GCN, IntentHead]:
    """
    Persist uploaded model binaries to disk (required for `torch.load`)
    and return instantiated, `.eval()`-ready model objects.

    Caching ensures models are only loaded once per session.
    """
    tmp_gnn = Path("_tmp_gnn.pt")
    tmp_head = Path("_tmp_head.pkl")
    tmp_gnn.write_bytes(gnn_bytes)
    tmp_head.write_bytes(head_bytes)

    gnn = GCN()
    gnn.load(tmp_gnn)  # custom `load` method inside GCN
    gnn.eval()

    head = IntentHead.load(tmp_head)
    return gnn, head


@st.cache_data(show_spinner="Computing embeddings & posteriors …")
def compute_emb_and_posteriors(
    df: pd.DataFrame, _gnn: GCN, _head: IntentHead
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a listings DataFrame into a DGL heterogeneous graph, obtain GNN
    embeddings for listing nodes, then predict calibrated posteriors.

    Returns
    -------
    emb : np.ndarray
        2-D array of node embeddings, index-aligned with `df`.
    prob : np.ndarray
        1-D array of posterior probabilities (P(best listing)).
    """
    g = ListingGraphTG().from_df(df, add_masks=False)
    with torch.no_grad():
        emb = _gnn.embed(g)[g.listing_idx].cpu().numpy()
        prob = _head.predict_proba(emb, df)
    return emb, prob


def explain_listing(
    head: IntentHead,
    emb_vec: np.ndarray,
    row: pd.Series,
    k: int = 10,
) -> pd.DataFrame:
    """
    Generate a top-k feature contribution table for a single listing.

    Returns a DataFrame with columns: ``feature``, ``Δ log-odds``.
    """
    feats = head.explain(emb_vec, row, top_k=k)
    return pd.DataFrame(feats, columns=["feature", "Δ log-odds"])
# ╰────────────────────────────────────────────────────────────────────╯


# ╭──────────────────── EARLY EXIT IF INPUTS MISSING ───────────────────╮
if not (gnn_file and head_file and csv_files):
    st.info("⬅️  Upload *both* models and at least one CSV to start.")
    st.stop()
# ╰─────────────────────────────────────────────────────────────────────╯


# ╭────────────── RESET SESSION HISTORY WHEN CSV SET CHANGES ───────────╮
csv_key: Tuple[str, ...] = tuple(sorted(f.name for f in csv_files))
if st.session_state.get("csv_key") != csv_key:
    st.session_state["csv_key"] = csv_key
    st.session_state["hist_scores"] = np.array([], dtype=float)
    st.session_state["prev_ids"]: Set[str] = set()
# ╰─────────────────────────────────────────────────────────────────────╯


# ╭───────────────────────── LOAD MODELS ONCE ──────────────────────────╮
gnn, head = load_models(gnn_file.read(), head_file.read())
# ╰─────────────────────────────────────────────────────────────────────╯


# Ensure chronological order (week0 → weekN)
csv_files.sort(key=lambda f: f.name)

hist_scores: np.ndarray = st.session_state["hist_scores"]
prev_ids: Set[str] = st.session_state["prev_ids"]
k_next: int = fixed_k  # seed for week 0

# ╭─────────────────────────── MAIN LOOP ───────────────────────────────╮
for w_idx, uploaded in enumerate(csv_files):
    # Each week gets its own collapsible panel
    with st.expander(f"📅 Week {w_idx} – {uploaded.name}", expanded=(w_idx == 0)):
        # 1) ── Load CSV ────────────────────────────────────────────────
        df = pd.read_csv(uploaded, na_values=["None", ""])
        if "sold_date" in df.columns:
            df["sold_date"] = pd.to_datetime(df["sold_date"], errors="coerce")

        # 2) ── Embeddings & posteriors ────────────────────────────────
        emb_mat, prob = compute_emb_and_posteriors(df, gnn, head)

        # Zero-out sold listings so they never trigger an offer
        if "sold_date" in df.columns:
            prob[df["sold_date"].notna().values] = 0.0

        # 3) ── Update posterior history for OVP fit ───────────────────
        hist_scores = np.concatenate([hist_scores, prob])
        st.session_state["hist_scores"] = hist_scores

        # 4) ── Determine k_next (expected arrivals) ────────────────────
        if auto_k and "listing_id" in df.columns:
            new_ids: Set[str] = set(df["listing_id"]) - prev_ids
            k_next = max(1, len(new_ids))
        else:
            k_next = fixed_k

        # Store IDs for next iteration
        if "listing_id" in df.columns:
            prev_ids = set(df["listing_id"])
            st.session_state["prev_ids"] = prev_ids

        # 5) ── Option-Value Planner decision ──────────────────────────
        ovp = (
            OptionValuePlanner(
                wait_cost=wait_cost,
                regret_cost=regret_cost,
                k_estimate=int(k_next),
            )
            .fit(hist_scores)
        )
        p_best: float = float(prob.max())
        ovp_stop, p_better, delta = ovp.should_stop(p_best)
        final_stop: bool = (p_best >= tau) and ovp_stop

        # 6) ── Table for display (sorted by posterior) ────────────────
        df_out = (
            df.assign(posterior=np.round(prob, 4), offer=prob >= tau)
            .sort_values("posterior", ascending=False)
        )

        # 7) ── Decision banner ────────────────────────────────────────
        banner = (
            "🟢 **RECOMMEND OFFER NOW**"
            if final_stop
            else "🟡 **WAIT – better listing likely next week**"
        )
        st.markdown(banner)
        st.caption(
            f"best p = {p_best:.3f} | "
            f"P(better) = {p_better:.1%} | "
            f"ΔRegret = {delta:+.4f} | "
            f"k_next = {k_next}"
        )

        # 8) ── Interactive data grid ─────────────────────────────────
        st.dataframe(
            df_out.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            column_order=list(df.columns) + ["posterior", "offer"],
            column_config={
                "posterior": st.column_config.NumberColumn(
                    "posterior",
                    format="%.4f",
                    help="Stacked-model probability",
                ),
                "offer": st.column_config.CheckboxColumn("offer"),
                "has_cladding": st.column_config.CheckboxColumn("has cladding"),
                "outdoor_space": st.column_config.CheckboxColumn("outdoor space"),
            },
        )

        # 9) ── Explainability panel ──────────────────────────────────
        st.divider()
        st.subheader("🔍 Explain a listing")

        sel = st.selectbox(
            "Choose row to explain (after current sort):",
            options=list(range(len(df_out))),
            format_func=lambda i: f"row {i} | p={df_out.iloc[i]['posterior']}",
        )

        # Map selection back to original DataFrame index
        raw_idx = df_out.iloc[int(sel)].name
        exp_df = explain_listing(head, emb_mat[raw_idx], df.iloc[raw_idx], k=8)
        st.dataframe(exp_df, hide_index=True, use_container_width=True)

        # 10) ── Download offers CSV ──────────────────────────────────
        offers_csv = df_out[df_out["offer"]].to_csv(index=False)
        st.download_button(
            f"📥 Download offers (Week {w_idx})",
            data=offers_csv,
            file_name=f"offers_week_{w_idx}.csv",
            mime="text/csv",
            key=f"dl_{w_idx}",
        )

    # ――― Early-exit: stop simulation once an offer is advised ―――
    if final_stop:
        st.success(f"Simulation halted — offer recommended in **Week {w_idx}**.")
        break
else:
    st.info("End of uploaded weeks — no offer triggered yet.")
