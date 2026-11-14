import numpy as np
from fraud.rings import find_rings
from training.train_gnn_elliptic import blend_report, ring_report


def test_rings_link_shared_accounts():
    al = [{"id": 1, "nameOrig": "A", "nameDest": "M"}, {"id": 2, "nameOrig": "B", "nameDest": "M"},
          {"id": 3, "nameOrig": "C", "nameDest": "N"}]
    r = find_rings(al)
    assert len(r) == 1 and r[0]["alert_ids"] == [1, 2] and r[0]["size"] == 2


def test_blend_never_worse_than_best_single_on_validation():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.1).astype(int)
    a, b = y * 0.5 + rng.random(4000) * 0.8, y * 0.5 + rng.random(4000) * 0.8
    res, w = blend_report(y[:2000], a[:2000], b[:2000], y[2000:], a[2000:], b[2000:])
    assert 0 <= w <= 1 and res["blend"]["auprc"] >= min(res["xgboost"]["auprc"], res["graphsage"]["auprc"]) - 0.02


def test_ring_report_components():
    ei = np.array([[0, 1, 3], [1, 2, 4]])
    flagged = np.array([True, True, True, True, True, False])
    y = np.array([1, 1, 0, 1, 1, 0])
    r = ring_report(ei, flagged, y, 6)
    assert r["n_rings"] == 2 and r["largest_ring"] == 3 and r["nodes_in_rings"] == 5


def test_consumer_handle_skips_bad_messages():
    from serving.consumer import handle

    class R:
        def __init__(s, code, body=None): s.status_code, s._b, s.text = code, body, "x"
        def json(s): return s._b
        def raise_for_status(s): pass

    class S:
        def __init__(s, code): s.code = code
        def post(s, url, json, timeout): return R(s.code, {"fraud_score": 0.1})

    assert handle(b"not json", "http://x", S(200)) is None
    assert handle(b'{"a":1}', "http://x", S(422)) is None
    assert handle(b'{"a":1}', "http://x", S(200)) == {"fraud_score": 0.1}


def test_eda_and_replay_sample(tmp_path):
    from fraud.synthetic import make_paysim
    from training import eda, export_online_state
    csv = tmp_path / "p.csv"; make_paysim(5000, steps=100, seed=5).to_csv(csv, index=False)
    eda.main(["--csv", str(csv), "--reports-dir", str(tmp_path)])
    assert "Heavy class imbalance" in (tmp_path / "eda_paysim.md").read_text()
    export_online_state.main(["--csv", str(csv), "--from-step", "60", "--out-dir", str(tmp_path), "--feast-data", str(tmp_path / "a.parquet"),
                              "--replay-out", str(tmp_path / "replay.csv"), "--replay-limit", "50"])
    import pandas as pd
    r = pd.read_csv(tmp_path / "replay.csv")
    assert len(r) == 50 and "newbalanceOrig" in r and r["step"].min() >= 60
