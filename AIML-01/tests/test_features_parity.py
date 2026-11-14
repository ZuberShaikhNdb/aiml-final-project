"""Training/serving parity: replaying transactions through the ONLINE path must reproduce the BATCH features exactly."""
import numpy as np
import pytest
from fraud.data import load_paysim
from fraud.features import FEATURES, batch_features, online_features, state_before, update_state
from fraud.store import MemoryStore
from fraud.synthetic import make_paysim


@pytest.fixture(scope="module")
def df(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "p.csv"
    make_paysim(12000, steps=120, seed=3).to_csv(p, index=False)
    return load_paysim(p)


def test_online_equals_batch(df):
    start = 80
    expected = batch_features(df)[df["step"] >= start]
    store = MemoryStore(state_before(df, start))
    rows = []
    for _, r in df[df["step"] >= start].iterrows():
        tx = r.to_dict()
        o, d = store.get(tx["nameOrig"]).get("o"), store.get(tx["nameDest"]).get("d")
        rows.append(online_features(tx, o, d))
        store.put(tx["nameOrig"], "o", update_state(o, tx["amount"], tx["step"]))
        store.put(tx["nameDest"], "d", update_state(d, tx["amount"], tx["step"]))
    got = np.array([[row[f] for f in FEATURES] for row in rows])
    np.testing.assert_allclose(got, expected[FEATURES].to_numpy(), rtol=1e-6, atol=1e-6, equal_nan=True)


def test_velocity_counts_window_and_cap(df):
    f = batch_features(df)
    assert f["d_vel_24h"].max() <= 50 and (f["d_vel_24h"] <= f["d_n_prior"]).all()
    assert f["d_n_prior"].max() > 1  # mule accounts are reused, so velocity features carry signal
