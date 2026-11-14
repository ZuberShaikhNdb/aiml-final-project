# Data

| Dataset | Used for | Source |
|---|---|---|
| PaySim synthetic mobile-money transactions (6.36M rows, 0.13% fraud) | EDA, XGBoost, Feast features, live replay | Kaggle: `ealaxi/paysim1` |
| Elliptic Bitcoin transaction graph (203k nodes, 234k edges, 49 time steps) | GraphSAGE, blend, ring detection | Kaggle: `ellipticco/elliptic-data-set` |

Raw data is not committed. `data/replay_sample.csv` is a 20,000-row slice of PaySim (later time steps) used for the demo replay.
`feature_repo/data/account_state.parquet` is the account state computed from earlier steps for the accounts in that slice.
