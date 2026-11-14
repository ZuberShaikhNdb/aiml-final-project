# Limits and honest caveats

**Educational project. Not a production fraud system.**

## Data
- **PaySim is synthetic.** It simulates mobile-money transactions from aggregated real logs. Fraud patterns are scripted, so scores on it say little about real-world performance. The balance-consistency features (`err_orig`, `err_dest`, `orig_emptied`, ...) are a well-known artefact of the simulator and make the task close to trivial. That is why we also train and report a **behavioural-only** model (no balance features) as the more honest number.
- **Only TRANSFER and CASH_OUT carry fraud** in PaySim, so those are the only types scored. Other types pass through with score 0 and the reason "not scored".
- **No device, location or card fields exist in PaySim.** The brief's "device and location changes" are not implemented. We built velocity (transactions in the last 24 steps), amount versus the account's own average, counterparty history and balance features instead.
- **Origin accounts almost never repeat in PaySim**, so origin-side velocity and averages are mostly empty; destination-side features carry the behavioural signal.
- **Elliptic features are anonymised** (165 columns), so the graph experiment cannot be interpreted feature by feature. Many nodes are unlabelled and are excluded from training and scoring.

## Models
- Single time-ordered split (train/validation/test by time). No cross-validation and no confidence intervals; one seed. Differences of a few thousandths between models are noise.
- The alert threshold maximises F1 on the validation period. A real team would pick it from analyst capacity and the cost of a miss versus a false alarm.
- Fraud rate differs a lot between periods: 0.19% in the PaySim training steps against 3.67% in the test steps. Test precision and AUPRC therefore reflect a richer fraud mix than training, and would be lower at the training base rate.
- On Elliptic, GraphSAGE scored 0.94 AUPRC on validation weeks and 0.55 on test weeks, and the blend gave no gain over XGBoost. The graph component did not help in these results.
- The GraphSAGE model is evaluated **offline on Elliptic only**. It is not in the live scoring path, because PaySim has too little shared structure to train a graph model, and serving a GNN needs graph state that this stack does not keep. Wiring it in would need a dataset with shared entities (devices, cards).

## System
- Single scoring replica with in-order processing, on purpose: per-account state must be updated in order. Scaling out needs partitioning by account.
- The online store is SQLite and the event log is SQLite. Fine for a demo, not for production volume.
- There is no authentication on the API or dashboard.
- Analyst feedback is collected but retraining does not merge it automatically.
- Kubernetes manifests target Minikube with locally built images and are meant for a demo cluster.
