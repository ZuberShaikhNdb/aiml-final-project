"""EDA on PaySim: class imbalance, fraud by type, amounts, counterparty reuse. Writes reports/eda_paysim.{json,md}.
python training/eda.py --csv data/paysim.csv"""
import argparse, json
from pathlib import Path
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--reports-dir", default="reports")
    a = ap.parse_args(argv)
    df = pd.read_csv(a.csv, usecols=["step", "type", "amount", "nameOrig", "nameDest", "isFraud"])
    sc = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]
    by_type = df.groupby("type")["isFraud"].agg(rows="size", fraud="sum")
    by_type["fraud_rate"] = by_type["fraud"] / by_type["rows"]
    fr = df[df["isFraud"] == 1]
    r = {"rows_total": int(len(df)), "fraud_total": int(df["isFraud"].sum()), "fraud_rate_all": float(df["isFraud"].mean()),
         "rows_scored_types": int(len(sc)), "fraud_rate_scored_types": float(sc["isFraud"].mean()),
         "fraud_by_type": by_type.reset_index().to_dict("records"),
         "median_amount_fraud": float(fr["amount"].median()), "median_amount_legit": float(df.loc[df["isFraud"] == 0, "amount"].median()),
         "unique_origin_accounts": int(df["nameOrig"].nunique()), "unique_dest_accounts": int(df["nameDest"].nunique()),
         "share_origins_seen_once": float((df["nameOrig"].value_counts() == 1).mean()),
         "share_dests_seen_once": float((df["nameDest"].value_counts() == 1).mean()),
         "fraud_per_step_mean": float(fr.groupby("step").size().mean()), "steps": int(df["step"].max())}
    out = Path(a.reports_dir); out.mkdir(exist_ok=True, parents=True)
    json.dump(r, open(out / "eda_paysim.json", "w"), indent=2)
    md = [f"# PaySim EDA", f"- Rows: {r['rows_total']:,}; fraud: {r['fraud_total']:,} ({r['fraud_rate_all']:.3%}). **Heavy class imbalance**: accuracy would be {1 - r['fraud_rate_all']:.2%} by predicting 'legit' always, so we use precision-recall AUC.",
          f"- Fraud only occurs in TRANSFER and CASH_OUT; within those types the fraud rate is {r['fraud_rate_scored_types']:.3%} ({r['rows_scored_types']:,} rows). The model is trained and served on these two types.",
          f"- Median amount: fraud {r['median_amount_fraud']:,.0f} vs legitimate {r['median_amount_legit']:,.0f}.",
          f"- Origin accounts seen only once: {r['share_origins_seen_once']:.1%}; destination accounts seen only once: {r['share_dests_seen_once']:.1%}. Few repeated counterparties is why the graph model is evaluated on Elliptic instead.",
          "", "| type | rows | fraud | fraud rate |", "|---|---|---|---|"] + \
         [f"| {x['type']} | {x['rows']:,} | {x['fraud']:,} | {x['fraud_rate']:.4%} |" for x in r["fraud_by_type"]]
    (out / "eda_paysim.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
