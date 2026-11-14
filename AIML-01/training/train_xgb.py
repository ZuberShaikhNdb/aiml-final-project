"""Train the real-time XGBoost model on PaySim and track the run in MLflow.
python training/train_xgb.py --csv data/paysim.csv --tag full
python training/train_xgb.py --csv data/paysim.csv --tag behavioural --no-balance-features"""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from fraud.data import load_paysim, split_masks
from fraud.features import BEHAVIOURAL, FEATURES, batch_features


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


def metrics_at(y, p, thr):
    pred = p >= thr
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"threshold": thr, "precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "tp": tp, "fp": fp, "fn": fn, "tn": int(((~pred) & (y == 0)).sum()), "alert_rate": float(pred.mean())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--no-balance-features", action="store_true")
    ap.add_argument("--train-end", type=int, default=450)
    ap.add_argument("--val-end", type=int, default=600)
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="weights")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    a = ap.parse_args(argv)
    feats = BEHAVIOURAL if a.no_balance_features else FEATURES

    df = load_paysim(a.csv)
    X, y = batch_features(df), df["isFraud"].to_numpy()
    tr, va, te = (m.to_numpy() for m in split_masks(df, a.train_end, a.val_end))
    print(f"rows train/val/test: {tr.sum()}/{va.sum()}/{te.sum()}  fraud rate train: {y[tr].mean():.4%}")

    model = xgb.XGBClassifier(n_estimators=a.n_estimators, max_depth=6, learning_rate=0.08, subsample=0.8,
                              colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=a.seed,
                              scale_pos_weight=float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)),  # class weights
                              eval_metric="aucpr", early_stopping_rounds=40)
    model.fit(X.loc[tr, feats], y[tr], eval_set=[(X.loc[va, feats], y[va])], verbose=50)

    pv, pt = model.predict_proba(X.loc[va, feats])[:, 1], model.predict_proba(X.loc[te, feats])[:, 1]
    thr = best_f1_threshold(y[va], pv)  # threshold chosen on validation only
    m = {"val_auprc": float(average_precision_score(y[va], pv)),
         "test_auprc": float(average_precision_score(y[te], pt)), "test_roc_auc": float(roc_auc_score(y[te], pt)),
         "test_fraud_rate": float(y[te].mean()), **{f"test_{k}": v for k, v in metrics_at(y[te], pt, thr).items()}}
    print(json.dumps(m, indent=2))

    out, rep = Path(a.out_dir), Path(a.reports_dir)
    out.mkdir(exist_ok=True, parents=True); rep.mkdir(exist_ok=True, parents=True)
    model.save_model(out / f"xgb_{a.tag}.json")
    meta = {"features": feats, "threshold": thr, "tag": a.tag, "train_end": a.train_end, "val_end": a.val_end,
            "balance_features": not a.no_balance_features, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "metrics": m}
    json.dump(meta, open(out / f"xgb_{a.tag}_meta.json", "w"), indent=2)
    json.dump(m, open(rep / f"xgb_{a.tag}_metrics.json", "w"), indent=2)
    X.loc[tr, feats].sample(min(20000, int(tr.sum())), random_state=a.seed).to_parquet(out / f"reference_{a.tag}.parquet")
    imp = sorted(zip(feats, model.feature_importances_.tolist()), key=lambda t: -t[1])
    json.dump(imp, open(rep / f"xgb_{a.tag}_importance.json", "w"), indent=2)

    import mlflow
    mlflow.set_tracking_uri(a.mlflow_uri)
    mlflow.set_experiment("fraud-xgboost")
    with mlflow.start_run(run_name=a.tag):
        mlflow.log_params({"tag": a.tag, "features": len(feats), "balance_features": not a.no_balance_features,
                           "train_end": a.train_end, "val_end": a.val_end, "seed": a.seed, "best_iteration": int(model.best_iteration)})
        mlflow.log_metrics({k: v for k, v in m.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(out / f"xgb_{a.tag}.json")); mlflow.log_artifact(str(out / f"xgb_{a.tag}_meta.json"))
    return m


if __name__ == "__main__":
    main()
