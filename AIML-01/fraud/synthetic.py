"""Synthetic PaySim-shaped data for TESTS and plumbing demos only. Never use it for reported metrics."""
import numpy as np
import pandas as pd


def make_paysim(n=30000, steps=240, seed=0, fraud_rate=0.01):
    rng = np.random.default_rng(seed)
    step = np.sort(rng.integers(1, steps + 1, n))
    typ = rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN"], n, p=[.2, .35, .3, .15])
    orig = np.array([f"C{x}" for x in rng.integers(0, n // 3, n)])
    dest = np.array([f"C{x}" for x in rng.integers(n, n + n // 4, n)])
    bal = rng.lognormal(9, 1.2, n)
    amount = np.minimum(bal * rng.uniform(.01, .6, n), bal)
    fraud = (np.isin(typ, ["TRANSFER", "CASH_OUT"])) & (rng.random(n) < fraud_rate * 2.5)
    mules = np.array([f"C{n + n // 4 + k}" for k in range(12)])  # a few mule accounts reused => rings
    dest[fraud] = rng.choice(mules, fraud.sum())
    amount[fraud] = bal[fraud]
    newo = np.where(fraud, 0.0, bal - amount)
    oldd = np.where(fraud, 0.0, rng.lognormal(9, 1.5, n))
    newd = np.where(fraud, 0.0, oldd + amount)
    return pd.DataFrame({"step": step, "type": typ, "amount": amount, "nameOrig": orig, "oldbalanceOrg": bal,
                         "newbalanceOrig": newo, "nameDest": dest, "oldbalanceDest": oldd, "newbalanceDest": newd,
                         "isFraud": fraud.astype(int), "isFlaggedFraud": 0})
