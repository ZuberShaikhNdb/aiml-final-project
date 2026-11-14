"""Replay PaySim transactions (arrival order) into Kafka, or straight to the API with --http.
python serving/producer.py --csv data/paysim.csv --from-step 601 --rate 50 --limit 20000"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fraud.data import load_paysim


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--from-step", type=int, default=601)
    ap.add_argument("--rate", type=float, default=50, help="transactions per second")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--bootstrap", default="localhost:9092")
    ap.add_argument("--topic", default="transactions")
    ap.add_argument("--http", default=None, help="post straight to this API base URL instead of Kafka (no Docker needed)")
    a = ap.parse_args(argv)
    df = load_paysim(a.csv)
    df = df[df["step"] >= a.from_step].head(a.limit)
    cols = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrg", "nameDest", "oldbalanceDest", "newbalanceDest"]
    if a.http:
        import requests
        send = lambda m: requests.post(a.http.rstrip("/") + "/score", json=m, timeout=30).raise_for_status()
        flush = lambda: None
    else:
        from confluent_kafka import Producer
        p = Producer({"bootstrap.servers": a.bootstrap})
        send = lambda m: (p.produce(a.topic, json.dumps(m).encode()), p.poll(0))
        flush = p.flush
    delay = 1.0 / a.rate if a.rate > 0 else 0
    for i, (_, r) in enumerate(df.iterrows(), 1):  # single partition + in-order sends keep account state consistent
        msg = {c: (r[c].item() if hasattr(r[c], "item") else r[c]) for c in cols}
        msg["true_label"] = int(r["isFraud"])
        send(msg)
        if i % 1000 == 0:
            print(f"sent {i}", flush=True)
        time.sleep(delay)
    flush()
    print(f"done: {len(df)} transactions")


if __name__ == "__main__":
    main()
