# AIML-01: Real-Time Financial Fraud Detection Platform

> Educational project on **simulated** data. Not a production fraud system. See [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

A system that reads a stream of transactions, scores each one for fraud risk in milliseconds, links suspicious accounts into fraud rings, monitors for data drift and gives analysts a dashboard to review cases and send back labels.

| Brief requirement | Where it is |
|---|---|
| Dataset + EDA, class imbalance | PaySim, `training/eda.py` -> `reports/eda_paysim.md` |
| Features (velocity, amount vs user average, ...) stored in Feast | `fraud/features.py`, `feature_repo/`, `fraud/store.py` |
| XGBoost with class weights, PR-AUC, MLflow | `training/train_xgb.py` |
| Graph neural network (PyTorch Geometric), blended with XGBoost | `training/train_gnn_elliptic.py` (offline, on Elliptic) |
| Kafka producer + FastAPI consumer service returning a fraud score | `serving/producer.py`, `serving/consumer.py`, `serving/api.py` |
| Evidently drift report + automatic retraining trigger | `monitoring/drift.py` |
| Analyst dashboard with alerts, case status, feedback labels | `dashboard/app.py` (Streamlit) |
| Kubernetes (Minikube) | `k8s/fraud-platform.yaml` |

**Two datasets, on purpose.** PaySim drives the live pipeline. The graph model is evaluated on the Elliptic Bitcoin graph (the brief's own suggestion for the GNN part) because PaySim accounts rarely repeat, so a graph model has little to learn there. The graph model is therefore an offline experiment, not part of the live scoring path. Details and other caveats are in the limitations doc.

## Architecture
Diagram and data flow: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Data sources: [docs/DATA.md](docs/DATA.md).

```
fraud/        shared code: features (batch + online), Feast/memory store, scoring, event log, rings
training/     eda.py, train_xgb.py, export_online_state.py, train_gnn_elliptic.py
serving/      FastAPI scoring service, Kafka producer and consumer
monitoring/   Evidently drift report and retraining trigger
dashboard/    Streamlit analyst dashboard
feature_repo/ Feast definitions and exported account state
k8s/          Minikube manifests
tests/        pytest suite
```

## Results

All numbers come from the training runs in `reports/` (seed 42, one run each).

**Live model: PaySim, TRANSFER + CASH_OUT, temporal split by step (1-450 train / 451-600 validation / 601-743 test).** The alert threshold was chosen on validation. The test period has 43,550 transactions, 1,600 of them fraud (3.67%), against 0.19% fraud in training, so precision and AUPRC here are measured on a much richer fraud mix than the model was trained on.

| Model | Test AUPRC | Precision | Recall | F1 | Test ROC AUC |
|---|---|---|---|---|---|
| XGBoost, all features | 0.9985 | 1.000 | 0.998 | 0.999 | 0.9999 |
| XGBoost, behavioural features only | 0.673 | 0.798 | 0.477 | 0.597 | 0.927 |

The all-features score is near perfect because PaySim's balance fields (`err_orig`, `err_dest`, `orig_emptied`, ...) almost reveal the label. That is an artefact of the simulator. The behavioural-only model, which uses velocity, amount-versus-average and counterparty history, is the more honest estimate of what the engineered features alone can do.

**Graph model: Elliptic Bitcoin, temporal split (1-29 / 30-34 / 35-49), illicit class, test period.**

| Model | AUPRC | Precision | Recall | F1 |
|---|---|---|---|---|
| XGBoost | 0.786 | 0.771 | 0.738 | 0.754 |
| GraphSAGE | 0.547 | 0.601 | 0.569 | 0.584 |
| Blend (95% XGBoost) | 0.785 | 0.769 | 0.738 | 0.753 |

GraphSAGE reached 0.94 AUPRC on the validation weeks but only 0.55 on the test weeks, and the blend put 95% of its weight on XGBoost, so **the graph model did not improve on XGBoost here**. Elliptic's later time steps are known to behave differently from earlier ones; we did not test whether that explains the drop. Ring detection (connected components among transactions the blend flags in the test period) found 1,116 candidate rings, largest 50 transactions; 80.0% of the labelled transactions inside rings are truly illicit. Most flagged transactions are unlabelled, so that figure covers only the labelled ones.

**Serving latency** (scoring API with Feast, per transaction): measured during the replay demo, see `/stats` and the dashboard header.

## Run it

### 1. Train (Kaggle, free CPU/GPU)
```bash
pip install -r requirements.txt
python training/eda.py --csv /kaggle/input/paysim1/PS_20174392719_1491204439457_log.csv
python training/train_xgb.py --csv <paysim.csv> --tag full
python training/train_xgb.py --csv <paysim.csv> --tag behavioural --no-balance-features
python training/export_online_state.py --csv <paysim.csv> --from-step 601
pip install torch_geometric
python training/train_gnn_elliptic.py --data-dir <elliptic folder>
```
Outputs: `weights/`, `reports/`, `feature_repo/data/account_state.parquet`, `data/replay_sample.csv`.

### 2. Run the whole platform (Docker)
```bash
docker compose up --build
docker compose --profile replay up producer      # streams the sample through Kafka
```
Dashboard http://localhost:8501, API docs http://localhost:8000/docs, MLflow http://localhost:5000.

### 3. Run without Docker (no Kafka; the producer posts straight to the API)
```bash
pip install -r requirements.txt
set MODEL_DIR=weights            # PowerShell: $env:MODEL_DIR="weights"
uvicorn serving.api:app --port 8000
python serving/producer.py --csv data/replay_sample.csv --from-step 0 --http http://localhost:8000 --rate 50 --limit 5000
streamlit run dashboard/app.py
```
To use Feast instead of the in-memory store: `cd feature_repo && feast apply && feast materialize-incremental 2030-01-01T00:00:00`, then start the API with `STORE=feast`.

### 4. Drift check and retraining trigger
```bash
python monitoring/drift.py --tag full --db events.db            # writes reports/drift_report.html
python monitoring/drift.py --tag full --db events.db --run-retrain --csv <paysim.csv>
```

### 5. Kubernetes (Minikube)
See the header comment in `k8s/fraud-platform.yaml`.

## Tests
```bash
pip install -r requirements.txt && pytest -q
```
The 11 tests use synthetic PaySim-shaped data and cover: online features match training features exactly, the API scores and logs events, alert cases and feedback, ring linking, malformed input, a non-scored type, Evidently detecting an injected shift, the consumer skipping bad messages, EDA and the replay export, and the graph blend and ring maths. They check the plumbing, not model accuracy. Accuracy comes from the training reports. The Feast store was also checked by hand against the real package (apply, materialize, read, write-back; materialized state identical to the in-memory store).

Not covered by automated tests: the Kafka producer/consumer against a live broker, Docker and Kubernetes deployment, and the GraphSAGE training loop (it needs PyTorch Geometric and a GPU).

## Reflection and presentation
Submitted separately as required by the brief.
