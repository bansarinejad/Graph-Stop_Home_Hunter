# 🏠 Graph-Stop Home Hunter

> End-to-end research prototype that turns raw real‑estate listings into **actionable “bid / wait” advice** with a hybrid Graph‑Neural‑Network, interpretable tabular head, and a Bayesian Option‑Value Planner.

&nbsp;

| Stage | Tech | Purpose |
|-------|------|---------|
| **Graph encoder** | 2‑layer **GAT** (swap‑able to GraphSAGE) | Learns structural context in a `buyer → listing → suburb` graph. |
| **Tabular head** | Gaussian NB **or** Logistic‑Regression + optional MI top‑k | Blends node embeddings with one‑hot features; delivers calibrated posteriors. |
| **Calibrator** | `CalibratedClassifierCV` (sigmoid / isotonic) | Makes `P(offer)` trustworthy. |
| **Decision layer** | **Option‑Value Planner (OVP)** | Balances weekly *wait‑cost* vs *regret* of missing a better listing. |
| **UI** | Streamlit | Interactive week‑by‑week simulation and per‑listing explainability. |

---

## 🌱 Project layout

```
.
├── graph_home_hunter.py        # ⬅ core CLI (train / predict / sweep / simulate)
├── home_hunter_app.py          # Streamlit UI
├── simulate_weeks.py           # Batch replay helper
├── create_mock_listings.py     # 10‑week snapshot generator
├── make_training_data_v4_fixed_live.py  # large train/test CSV generator
├── option_value.py             # standalone OVP prototype
└── data_snapshots/             # ⇡ auto‑generated weekly CSVs
```

---

## 🚀 Quick‑start (CPU, Python ≥ 3.9)

```bash
git clone https://github.com/your-org/graph-stop-home-hunter.git
cd graph-stop-home-hunter
python -m venv venv && source venv/bin/activate      # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

Generate synthetic data and train everything:

```bash
# 1) offline data
python make_training_data_v4_fixed_live.py      # ⇢ train_listings.csv / test_listings.csv
python create_mock_listings.py                  # ⇢ data_snapshots/week_0.csv … week_9.csv

# 2) train GNN
python graph_home_hunter.py        --train-graph train_listings.csv gnn.pt

# 3) train stack (embeddings + tabular head)
python graph_home_hunter.py        --train-stack train_listings.csv gnn.pt head.pkl        --stack-model logreg --calib isotonic --top-k-tab 150
```

Run a one‑off week:

```bash
python graph_home_hunter.py        --predict-stack data_snapshots/week_5.csv gnn.pt head.pkl        --thr 0.95 --wait-cost 0.02 --regret-cost 0.35
```

Replay many weeks with the Bayesian planner:

```bash
python simulate_weeks.py                                             data_snapshots/week_5.csv data_snapshots/week_6.csv ...        --gnn gnn.pt --head head.pkl --thr 0.95
```

Launch the Streamlit demo:

```bash
streamlit run home_hunter_app.py
```

---

## 📊 Key results (synthetic‑data baseline)

| Model | ROC‑AUC | F1 @ τ=0.95 | *Median wait weeks* |
|-------|---------|-------------|---------------------|
| GAT‑only | 0.81 | 0.57 | 4 |
| **GAT + LogReg + OVP** | **0.89** | **0.71** | **1** |

*(seed 42, 20 k training rows, 10‑week simulation)*

---

## 🔍 Explainability

* `IntentHead.explain()` returns **Δ log‑odds** per feature.  
* The UI surfaces the top‑k drivers so users see *why* a listing is (or isn’t) recommended.

![explain](docs/img/explainability_example.png)

---

## ⚙️ Hyper‑parameters that matter

| Flag | Effect |
|------|--------|
| `--top-k-tab` | keep only the highest‑MI one‑hot columns (0 = all) |
| `--calib` | `sigmoid` (Platt) or `isotonic` calibration |
| `--thr` | posterior threshold to trigger an *offer candidate* |
| `--wait-cost`, `--regret-cost`, `--k-next-week` | behaviour of the Option‑Value Planner |

---

## 🛠️ TODO / ideas

* Edge‑typed HeteroGNN for buyer‑listing vs listing‑suburb edges  
* Thompson‑sampling to adapt OVP parameters online  
* Hook into a live Domain / Zillow feed  

PRs welcome!

---

## 📝 License

MIT – have fun, give credit, no warranty.
