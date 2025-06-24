### 📄 Repository description (one-liner)

> **Graph-Stop Home-Hunter** – a hybrid GNN + tabular pipeline that learns to spot the best property listings and decides *when* to bid using a Bayesian option-value planner.

---

## README.md

```markdown
# 🏠 Graph-Stop Home-Hunter

> End-to-end research prototype that turns raw real-estate listings into **actionable “bid / wait” advice** with a hybrid Graph-Neural-Network, interpretable tabular head, and a Bayesian Option-Value Planner.

&nbsp;

| Stage | Tech | Purpose |
|-------|------|---------|
| **Graph encoder** | 2-layer **GAT** (swap-able to GraphSAGE) | Learns structural context in a `buyer → listing → suburb` graph. |
| **Tabular head** | Gaussian NB **or** Logistic-Regression + optional MI top-k | Blends node embeddings with one-hot features; delivers calibrated posteriors. |
| **Calibrator** | `CalibratedClassifierCV` (sigmoid / isotonic) | Makes `P(offer)` trustworthy. |
| **Decision layer** | **Option-Value Planner (OVP)** | Balances weekly *wait-cost* vs *regret* of missing a better listing. |
| **UI** | Streamlit | Interactive week-by-week simulation and per-listing explainability. |

---

## 🌱 Project layout

```

.
├── graph\_home\_hunter.py        # ⬅️ core CLI (train / predict / sweep / simulate)
├── home\_hunter\_app.py          # Streamlit UI
├── simulate\_weeks.py           # Batch replay helper
├── create\_mock\_listings.py     # 10-week snapshot generator
├── make\_training\_data\_v4\_fixed\_live.py  # large train/test CSV generator
├── option\_value.py             # standalone OVP prototype
└── data\_snapshots/             # ⇡ auto-generated weekly CSVs

````

---

## 🚀 Quick-start (CPU, Python ≥3.9)

```bash
git clone https://github.com/your-org/graph-stop-home-hunter.git
cd graph-stop-home-hunter
python -m venv venv && source venv/bin/activate          # or `.\venv\Scripts\activate`
pip install -r requirements.txt
````

Generate synthetic data and train everything:

```bash
# 1) offline data
python make_training_data_v4_fixed_live.py          # ⇢ train_listings.csv / test_listings.csv
python create_mock_listings.py                      # ⇢ data_snapshots/week_0.csv … week_9.csv

# 2) train GNN
python graph_home_hunter.py \
       --train-graph train_listings.csv gnn.pt

# 3) train stack (embeddings + tabular head)
python graph_home_hunter.py \
       --train-stack train_listings.csv gnn.pt head.pkl \
       --stack-model logreg --calib isotonic --top-k-tab 150
```

Run a one-off week:

```bash
python graph_home_hunter.py \
       --predict-stack data_snapshots/week_5.csv gnn.pt head.pkl \
       --thr 0.95 --wait-cost 0.02 --regret-cost 0.35
```

Replay many weeks with the Bayesian planner:

```bash
python simulate_weeks.py                                      \
       data_snapshots/week_5.csv data_snapshots/week_6.csv ... \
       --gnn gnn.pt --head head.pkl --thr 0.95
```

Launch the Streamlit demo:

```bash
streamlit run home_hunter_app.py
```

---

## 📊 Key results (synthetic-data baseline)

| Model                  | ROC-AUC  | F1 @ τ=0.95 | *Median wait weeks* |
| ---------------------- | -------- | ----------- | ------------------- |
| GAT-only               | 0.81     | 0.57        | 4                   |
| **GAT + LogReg + OVP** | **0.89** | **0.71**    | **1**               |

*(seed 42, 20 k training rows, 10-week simulation)*

---

## 🔍 Explainability

* `IntentHead.explain()` returns **Δ log-odds** per feature.
* The UI surfaces the top-k drivers so users see *why* a listing is (or isn’t) recommended.

<p align="center"><img src="docs/img/explainability_example.png" width="480"></p>

---

## ⚙️ Hyper-parameters that matter

| Flag                                            | Effect                                                      |
| ----------------------------------------------- | ----------------------------------------------------------- |
| `--top-k-tab`                                   | keep only the highest-MI one-hot columns (0 = all)          |
| `--calib`                                       | `sigmoid` (Platt) or `isotonic` for probability calibration |
| `--thr`                                         | posterior threshold to trigger an *offer candidate*         |
| `--wait-cost`, `--regret-cost`, `--k-next-week` | behaviour of the Option-Value Planner                       |

---

## 📑 Citation

If you use *Graph-Stop Home-Hunter* in academic work:

```text
@misc{graphstophunter2025,
  author  = {Your Name},
  title   = {Graph-Stop Home Hunter: Intent-aware decision making with GNNs},
  year    = {2025},
  url     = {https://github.com/your-org/graph-stop-home-hunter}
}
```

---

## 🛠️ TODO / ideas

* Edge-typed HeteroGNN to differentiate buyer-listing vs listing-suburb.
* Online reinforcement of the OVP with Thompson sampling.
* Real-world feed via Domain or Zillow API.

Pull requests welcome!

---

## 📝 License

MIT – have fun, give credit, no warranty.

```

> **What “one-hot tabular” means:** every categorical column (e.g. `condition`, `suburb`, `sold_date NA/​not-NA`) is expanded into a 0/1 **one-hot vector**. Those vectors are concatenated with the 128-dim GNN embeddings and fed into the head classifier.
```
