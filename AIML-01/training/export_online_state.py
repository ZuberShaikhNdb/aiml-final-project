"""Compute account state as of the replay start step and publish it for serving (memory pickle + Feast parquet).
python training/export_online_state.py --csv data/paysim.csv --from-step 601"""
import argparse, pickle, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from fraud.data import load_paysim
from fraud.features import state_before


def to_frame(state):
    rows = []
    for acct, roles in state.items():
        row = {"account_id": acct, "event_timestamp": datetime.now(timezone.utc)}
        for r in ("o", "d"):
            s = roles.get(r, {"n": 0, "sum": 0.0, "recent": []})
            row[f"{r}_n"], row[f"{r}_sum"], row[f"{r}_recent"] = int(s["n"]), float(s["sum"]), [int(x) for x in s["recent"]]
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--from-step", type=int, default=601)
    ap.add_argument("--out-dir", default="weights")
    ap.add_argument("--feast-data", default="feature_repo/data/account_state.parquet")
    ap.add_argument("--replay-out", default="data/replay_sample.csv", help="small PaySim slice (arrival order) used by the demo replay")
    ap.add_argument("--replay-limit", type=int, default=20000)
    a = ap.parse_args(argv)
    df = load_paysim(a.csv)
    state = state_before(df, a.from_step)
    Path(a.replay_out).parent.mkdir(exist_ok=True, parents=True)
    sample = df[df["step"] >= a.from_step].head(a.replay_limit)
    sample = sample.drop(columns=[c for c in ("newbalanceOrig",) if c in sample]).rename(columns={"newbalanceOrg": "newbalanceOrig"})
    sample.to_csv(a.replay_out, index=False)
    Path(a.out_dir).mkdir(exist_ok=True, parents=True)
    pickle.dump(state, open(Path(a.out_dir) / "online_state.pkl", "wb"))
    Path(a.feast_data).parent.mkdir(exist_ok=True, parents=True)
    to_frame(state).to_parquet(a.feast_data)
    print(f"exported state for {len(state)} accounts (as of step < {a.from_step}); replay sample: {len(sample)} rows -> {a.replay_out}")


if __name__ == "__main__":
    main()
