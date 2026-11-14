"""End to end: train on synthetic data, serve through the API, review a case, check rings and drift."""
import importlib, json, sys
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from fraud.synthetic import make_paysim


@pytest.fixture(scope="module")
def env(tmp_path_factory, monkeypatch_module=None):
    d = tmp_path_factory.mktemp("run")
    csv = d / "p.csv"
    make_paysim(40000, steps=240, seed=1).to_csv(csv, index=False)
    from training import train_xgb, export_online_state
    m = train_xgb.main(["--csv", str(csv), "--train-end", "120", "--val-end", "170", "--n-estimators", "150",
                        "--out-dir", str(d / "w"), "--reports-dir", str(d / "r"), "--mlflow-uri", f"sqlite:///{d}/mlflow.db"])
    export_online_state.main(["--csv", str(csv), "--from-step", "171", "--out-dir", str(d / "w"), "--feast-data", str(d / "acct.parquet")])
    return d, csv, m


@pytest.fixture(scope="module")
def client(env, monkeypatch_module):
    d, _, _ = env
    monkeypatch_module.setenv("MODEL_DIR", str(d / "w"))
    monkeypatch_module.setenv("DB_PATH", str(d / "events.db"))
    sys.modules.pop("serving.api", None)
    api = importlib.import_module("serving.api")
    return TestClient(api.app)


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch(); yield mp; mp.undo()


def test_model_learns_synthetic_signal(env):
    assert env[2]["test_auprc"] > 0.5  # plumbing check only; says nothing about real-world accuracy


def test_score_alerts_cases_rings(env, client):
    _, csv, _ = env
    df = pd.read_csv(csv)
    df = df[(df["step"] >= 171) & df["type"].isin(["TRANSFER", "CASH_OUT"])]
    assert client.get("/health").json()["status"] == "ok"
    for _, r in df.head(1500).iterrows():
        body = {k: (r[k].item() if hasattr(r[k], "item") else r[k]) for k in
                ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "nameDest", "oldbalanceDest", "newbalanceDest"]}
        body["newbalanceOrg"], body["true_label"] = float(r["newbalanceOrig"]), int(r["isFraud"])
        res = client.post("/score", json=body)
        assert res.status_code == 200
    out = res.json()
    assert 0 <= out["fraud_score"] <= 1 and out["decision"] in ("alert", "pass") and len(out["reasons"]) == 3
    stats = client.get("/stats").json()
    assert stats["n"] == 1500 and stats["alerts"] > 0 and stats["caught"] > 0
    a = client.get("/alerts").json()[0]
    assert client.post(f"/alerts/{a['id']}/case", json={"status": "confirmed_fraud", "label": 1, "analyst": "t"}).status_code == 200
    assert client.post(f"/alerts/{a['id']}/case", json={"status": "bogus"}).status_code == 422
    assert client.post("/alerts/999999/case", json={"status": "investigating"}).status_code == 404
    assert len(client.get("/feedback").json()) == 1
    assert len(client.get("/rings").json()) >= 1  # synthetic mule accounts are shared across alerts


def test_non_scored_type_and_bad_input(client):
    base = dict(step=200, amount=10.0, nameOrig="C1", oldbalanceOrg=100.0, newbalanceOrg=90.0, nameDest="M1", oldbalanceDest=0.0, newbalanceDest=0.0)
    r = client.post("/score", json={**base, "type": "PAYMENT"}).json()
    assert r["decision"] == "not_scored" and r["fraud_score"] == 0
    assert client.post("/score", json={"type": "TRANSFER"}).status_code == 422


def test_drift_detects_shift(env, client):
    from monitoring import drift
    d, _, _ = env
    ref = pd.read_parquet(d / "w" / "reference_full.parquet")
    same = ref.sample(1000, random_state=0)
    shifted = ref.sample(1000, random_state=1).copy()
    for c in ["amount_log1p", "d_n_prior", "o_vel_24h", "d_vel_24h", "err_orig", "oldbalanceOrg"]:
        shifted[c] = shifted[c] * 5 + 50
    assert drift.drift_summary(ref, same)["share_drifted"] < 0.2
    s = drift.drift_summary(ref, shifted, d / "drift.html")
    assert s["share_drifted"] >= 0.3 and (d / "drift.html").exists()
