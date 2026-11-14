"""Fraud-ring detection on the Elliptic Bitcoin graph: GraphSAGE + XGBoost, blended.
Runs on Kaggle (GPU): pip install torch_geometric
python training/train_gnn_elliptic.py --data-dir /kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset
Temporal split (Elliptic has 49 time steps): train 1-29, validation 30-34, test 35-49."""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def load_elliptic(d):
    d = Path(d)
    feats = pd.read_csv(d / "elliptic_txs_features.csv", header=None)
    classes = pd.read_csv(d / "elliptic_txs_classes.csv")
    edges = pd.read_csv(d / "elliptic_txs_edgelist.csv")
    ids = feats[0].to_numpy()
    idx = {t: i for i, t in enumerate(ids)}
    cls = classes.set_index("txId").loc[ids, "class"].astype(str).to_numpy()
    y = np.where(cls == "1", 1, np.where(cls == "2", 0, -1))  # 1 illicit, 0 licit, -1 unknown
    e = np.array([[idx[a], idx[b]] for a, b in edges.itertuples(index=False) if a in idx and b in idx]).T
    return feats[1].to_numpy(), feats.iloc[:, 2:].to_numpy(np.float32), y, e


def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


def scores(y, p, thr):
    pred = p >= thr
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"auprc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
            "precision": pr, "recall": rc, "f1": 2 * pr * rc / max(pr + rc, 1e-12), "threshold": thr}


def blend_report(yv, pv_x, pv_g, yt, pt_x, pt_g):
    """Pick the blend weight on validation, report all three models on test. Pure numpy: unit-tested."""
    grid = np.linspace(0, 1, 21)
    w = float(grid[int(np.argmax([average_precision_score(yv, x * pv_x + (1 - x) * pv_g) for x in grid]))])
    out = {"blend_weight_xgb": w}
    for name, pv, pt in (("xgboost", pv_x, pt_x), ("graphsage", pv_g, pt_g), ("blend", w * pv_x + (1 - w) * pv_g, w * pt_x + (1 - w) * pt_g)):
        out[name] = scores(yt, pt, best_threshold(yv, pv))
    return out, w


def ring_report(edge_index, flagged, y, n_nodes):
    """Connected components among flagged nodes = candidate fraud rings (size >= 2)."""
    src, dst = edge_index
    keep = flagged[src] & flagged[dst]
    g = coo_matrix((np.ones(keep.sum()), (src[keep], dst[keep])), shape=(n_nodes, n_nodes))
    _, lab = connected_components(g, directed=False)
    comp = pd.DataFrame({"c": lab[flagged], "y": y[flagged]})
    sizes = comp.groupby("c").size()
    rings = sizes[sizes >= 2].index
    inr = comp[comp["c"].isin(rings)]
    lbl = inr[inr["y"] >= 0]
    return {"n_flagged": int(flagged.sum()), "n_rings": int(len(rings)), "largest_ring": int(sizes.max()) if len(sizes) else 0,
            "nodes_in_rings": int(len(inr)), "ring_precision_labelled": float((lbl["y"] == 1).mean()) if len(lbl) else None}


def train_graphsage(x, y, edge_index, tr, va, hidden, epochs, seed, dev):
    import torch, torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    from torch_geometric.utils import to_undirected
    torch.manual_seed(seed)

    class Net(torch.nn.Module):
        def __init__(s, i, h):
            super().__init__()
            s.c1, s.c2, s.out = SAGEConv(i, h), SAGEConv(h, h), torch.nn.Linear(h, 2)

        def forward(s, x, ei):
            x = F.dropout(F.relu(s.c1(x, ei)), 0.5, s.training)
            return s.out(F.relu(s.c2(x, ei)))

    X = torch.tensor(x, device=dev)
    ei = to_undirected(torch.tensor(edge_index, dtype=torch.long, device=dev))
    Y = torch.tensor(y, device=dev)
    tri, vai = torch.tensor(np.where(tr)[0], device=dev), torch.tensor(np.where(va)[0], device=dev)
    pos = float((y[tr] == 1).sum())
    w = torch.tensor([1.0, float((y[tr] == 0).sum()) / max(pos, 1)], device=dev).clamp(max=20)  # class weights
    net = Net(x.shape[1], hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=5e-4)
    best, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        net.train(); opt.zero_grad()
        loss = F.cross_entropy(net(X, ei)[tri], Y[tri], weight=w)
        loss.backward(); opt.step()
        if ep % 5 == 0:
            net.eval()
            with torch.no_grad():
                pv = torch.softmax(net(X, ei), 1)[:, 1]
            ap = average_precision_score(y[va], pv[vai].cpu().numpy())
            if ap > best:
                best, best_state = ap, {k: v.clone() for k, v in net.state_dict().items()}
            if ep % 50 == 0:
                print(f"epoch {ep} loss={loss.item():.4f} val_auprc={ap:.4f}", flush=True)
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        return torch.softmax(net(X, ei), 1)[:, 1].cpu().numpy(), best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="weights")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    a = ap.parse_args(argv)
    t, x, y, ei = load_elliptic(a.data_dir)
    lab = y >= 0
    tr, va, te = lab & (t <= 29), lab & (t >= 30) & (t <= 34), lab & (t >= 35)
    print(f"nodes={len(y)} edges={ei.shape[1]} labelled train/val/test={tr.sum()}/{va.sum()}/{te.sum()} illicit rate train={y[tr].mean():.3%}")

    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
    xs = ((x - mu) / sd).astype(np.float32)
    gb = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                           scale_pos_weight=float((y[tr] == 0).sum() / (y[tr] == 1).sum()), random_state=a.seed, n_jobs=-1)
    gb.fit(x[tr], y[tr])
    px = gb.predict_proba(x)[:, 1]

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pg, best_val = train_graphsage(xs, y, ei, tr, va, a.hidden, a.epochs, a.seed, dev)

    res, w = blend_report(y[va], px[va], pg[va], y[te], px[te], pg[te])
    pb = w * px + (1 - w) * pg
    flagged = (pb >= res["blend"]["threshold"]) & (t >= 35)
    res["rings_test_period"] = ring_report(ei, flagged, y, len(y))
    res.update({"dataset": "Elliptic Bitcoin (temporal split 1-29 / 30-34 / 35-49)", "device": dev, "seed": a.seed,
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum())})
    Path(a.reports_dir).mkdir(exist_ok=True, parents=True)
    json.dump(res, open(Path(a.reports_dir) / "gnn_blend_metrics.json", "w"), indent=2)
    rows = [f"| {m} | {res[m]['auprc']:.3f} | {res[m]['precision']:.3f} | {res[m]['recall']:.3f} | {res[m]['f1']:.3f} |" for m in ("xgboost", "graphsage", "blend")]
    (Path(a.reports_dir) / "gnn_blend_metrics.md").write_text("| Model (test, illicit class) | AUPRC | Precision | Recall | F1 |\n|---|---|---|---|---|\n" + "\n".join(rows))
    print(json.dumps(res, indent=2))

    import mlflow
    mlflow.set_tracking_uri(a.mlflow_uri); mlflow.set_experiment("fraud-gnn-elliptic")
    with mlflow.start_run(run_name="graphsage+xgb"):
        mlflow.log_params({"hidden": a.hidden, "epochs": a.epochs, "seed": a.seed, "blend_weight_xgb": w})
        mlflow.log_metrics({f"test_{m}_{k}": res[m][k] for m in ("xgboost", "graphsage", "blend") for k in ("auprc", "precision", "recall", "f1")})
    return res


if __name__ == "__main__":
    main()
