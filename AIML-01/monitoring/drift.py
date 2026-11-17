"""Data-drift report with Evidently + automatic retraining trigger.
python monitoring/drift.py --tag full --db events.db [--run-retrain --csv data/paysim.csv]"""
import argparse, json, sqlite3, subprocess, sys
from pathlib import Path
import pandas as pd


def drift_summary(reference: pd.DataFrame, current: pd.DataFrame, html_path=None):
    """Returns {'share_drifted': float, 'drifted_columns': [...], 'n_columns': int}."""
    try:  # Evidently >= 0.7
        from evidently import Report
        from evidently.presets import DataDriftPreset
        snap = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
        d = snap.dict()
        drifted, n, evidently_count = [], 0, None
        for m in d["metrics"]:
            cfg = m.get("config", {})
            if "column" in cfg:  # one ValueDrift metric per column
                n += 1
                thr, val = cfg.get("threshold", 0.1), float(m["value"])
                # p-value tests (K-S, chi-square, ...) flag drift when the value is BELOW the threshold;
                # distance scores (Wasserstein, Jensen-Shannon, ...) flag drift when it is at or above it.
                if (val < thr) if "p_value" in str(cfg.get("method", "")) else (val >= thr):
                    drifted.append(cfg["column"])
            elif isinstance(m.get("value"), dict) and "count" in m["value"]:
                evidently_count = int(m["value"]["count"])  # Evidently's own drifted-column count
        if evidently_count is not None and evidently_count != len(drifted):
            raise RuntimeError(f"drift parsing disagrees with Evidently: {len(drifted)} vs {evidently_count}")
        if html_path:
            snap.save_html(str(html_path))
        n = max(n, 1)
    except ImportError:  # legacy Evidently 0.4.x
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
        rep = Report(metrics=[DataDriftPreset()])
        rep.run(reference_data=reference, current_data=current)
        d = rep.as_dict()["metrics"][0]["result"]
        drifted = [c for c, v in d["drift_by_columns"].items() if v["drift_detected"]]
        n = d["number_of_columns"]
        if html_path:
            rep.save_html(str(html_path))
    return {"share_drifted": len(drifted) / n, "drifted_columns": drifted, "n_columns": n}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--db", default="events.db")
    ap.add_argument("--window", type=int, default=5000)
    ap.add_argument("--threshold", type=float, default=0.3, help="retrain if this share of features drifted")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--run-retrain", action="store_true")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args(argv)
    ref = pd.read_parquet(Path(a.weights) / f"reference_{a.tag}.parquet")
    con = sqlite3.connect(a.db)
    cur = pd.DataFrame([json.loads(r[0]) for r in con.execute("SELECT features FROM events ORDER BY id DESC LIMIT ?", (a.window,))])
    if len(cur) < 200:
        print("not enough scored transactions yet for a drift check"); return None
    cur = cur[list(ref.columns)]
    Path(a.reports_dir).mkdir(exist_ok=True, parents=True)
    s = drift_summary(ref, cur, Path(a.reports_dir) / "drift_report.html")
    s["retrain_triggered"] = s["share_drifted"] >= a.threshold
    json.dump(s, open(Path(a.reports_dir) / "drift_summary.json", "w"), indent=2)
    print(json.dumps(s, indent=2))
    if s["retrain_triggered"]:
        Path(a.reports_dir, "retrain_requested.flag").write_text("drift threshold exceeded")
        if a.run_retrain and a.csv:
            subprocess.run([sys.executable, "training/train_xgb.py", "--csv", a.csv, "--tag", a.tag], check=True)
    return s


if __name__ == "__main__":
    main()
