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
python make_training_data.py                # ⇢ train_listings.csv / test_listings.csv
python create_mock_listings.py              # ⇢ data_snapshots/week_0.csv … week_9.csv

# 2) train GNN
python graph_home_hunter.py --train-graph train_graph.pt gnn_v1.pt --epochs 150 --patience 50 --lr 1e-3

# 3) train stack (embeddings + tabular head)
python graph_home_hunter.py --train-stack train_listings.csv gnn_v1.pt logreg_head.pkl --stack-model logreg --calib sigmoid
```

Replay many weeks with the Bayesian planner:

```bash
python simulate_weeks.py --gnn gnn_v1.pt --head logreg_head.pkl --thr 0.95 --wait-cost 0.02 --regret-cost 0.2 week_0.csv week_1.csv week_2.csv week_3.csv week_4.csv week_5.csv week_6.csv week_7.csv week_8.csv week_9.csv
```

Launch the Streamlit demo:

```bash
streamlit run home_hunter_app.py
```

---

## 🔍 Explainability

* `IntentHead.explain()` returns **Δ log‑odds** per feature.  
* The UI surfaces the top‑k drivers so users see *why* a listing is (or isn’t) recommended.
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
