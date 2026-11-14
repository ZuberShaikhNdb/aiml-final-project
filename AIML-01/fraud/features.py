"""Feature definitions shared by training (batch) and serving (online).
One definition of every feature means no train/serve skew; tests/test_features_parity.py enforces it."""
import numpy as np
import pandas as pd

SCORED_TYPES = ("TRANSFER", "CASH_OUT")  # in PaySim, fraud only ever occurs in these two types
WINDOW = 24       # velocity window, in steps (1 step = 1 hour)
RECENT_CAP = 50   # max recent timestamps kept per account and role

BEHAVIOURAL = ["amount_log1p", "is_cash_out", "hour", "amt_vs_orig_mean", "amt_vs_dest_mean",
               "o_n_prior", "d_n_prior", "o_vel_24h", "d_vel_24h"]
BALANCE = ["oldbalanceOrg", "newbalanceOrg", "oldbalanceDest", "newbalanceDest",
           "err_orig", "err_dest", "orig_emptied", "dest_empty_before"]
FEATURES = BEHAVIOURAL + BALANCE
EMPTY = {"n": 0, "sum": 0.0, "recent": []}


# ---------------- online (one transaction at a time) ----------------
def _ratio(amount, st):
    if st["n"] <= 0:
        return np.nan
    mean = st["sum"] / st["n"]
    return amount / mean if mean > 0 else np.nan


def _vel(st, step):
    return float(min(sum(1 for s in st["recent"] if step - s <= WINDOW), RECENT_CAP))


def online_features(tx, o_state, d_state):
    """tx: dict with PaySim fields. o_state / d_state: account state as origin / as destination (or None)."""
    o, d = o_state or EMPTY, d_state or EMPTY
    amt, step = float(tx["amount"]), int(tx["step"])
    return {
        "amount_log1p": float(np.log1p(amt)),
        "is_cash_out": float(tx["type"] == "CASH_OUT"),
        "hour": float(step % 24),
        "amt_vs_orig_mean": _ratio(amt, o),
        "amt_vs_dest_mean": _ratio(amt, d),
        "o_n_prior": float(o["n"]), "d_n_prior": float(d["n"]),
        "o_vel_24h": _vel(o, step), "d_vel_24h": _vel(d, step),
        "oldbalanceOrg": float(tx["oldbalanceOrg"]), "newbalanceOrg": float(tx["newbalanceOrg"]),
        "oldbalanceDest": float(tx["oldbalanceDest"]), "newbalanceDest": float(tx["newbalanceDest"]),
        "err_orig": float(tx["newbalanceOrg"]) + amt - float(tx["oldbalanceOrg"]),
        "err_dest": float(tx["oldbalanceDest"]) + amt - float(tx["newbalanceDest"]),
        "orig_emptied": float(float(tx["newbalanceOrg"]) == 0 and float(tx["oldbalanceOrg"]) > 0),
        "dest_empty_before": float(float(tx["oldbalanceDest"]) == 0 and float(tx["newbalanceDest"]) == 0),
    }


def update_state(st, amount, step):
    st = st or EMPTY
    return {"n": st["n"] + 1, "sum": st["sum"] + float(amount),
            "recent": (list(st["recent"]) + [int(step)])[-RECENT_CAP:]}


# ---------------- batch (whole table, same maths, vectorised) ----------------
def _window_counts(codes, steps):
    n = len(codes)
    order = np.lexsort((np.arange(n), steps, codes))
    key = codes[order].astype(np.int64) * 1_000_000 + steps[order].astype(np.int64)
    lb = np.searchsorted(key, key - WINDOW, side="left")
    res = np.empty(n, dtype=np.float64)
    res[order] = np.minimum(np.arange(n) - lb, RECENT_CAP)
    return res


def batch_features(df):
    """df: rows in ARRIVAL ORDER (step, then file order), TRANSFER/CASH_OUT only."""
    df = df.reset_index(drop=True)
    amt = df["amount"].astype("float64")
    steps = df["step"].to_numpy(dtype=np.int64)
    out = pd.DataFrame(index=df.index)
    out["amount_log1p"] = np.log1p(amt)
    out["is_cash_out"] = (df["type"] == "CASH_OUT").astype("float64")
    out["hour"] = (df["step"] % 24).astype("float64")
    for role, col, name in (("o", "nameOrig", "orig"), ("d", "nameDest", "dest")):
        g = df.groupby(col, sort=False)
        n = g.cumcount().astype("float64")
        mean = (g["amount"].cumsum() - amt) / n.where(n > 0)
        out[f"amt_vs_{name}_mean"] = amt / mean.where(mean > 0)
        out[f"{role}_n_prior"] = n
        out[f"{role}_vel_24h"] = _window_counts(pd.factorize(df[col])[0], steps)
    for c in ("oldbalanceOrg", "newbalanceOrg", "oldbalanceDest", "newbalanceDest"):
        out[c] = df[c].astype("float64")
    out["err_orig"] = out["newbalanceOrg"] + amt - out["oldbalanceOrg"]
    out["err_dest"] = out["oldbalanceDest"] + amt - out["newbalanceDest"]
    out["orig_emptied"] = ((out["newbalanceOrg"] == 0) & (out["oldbalanceOrg"] > 0)).astype("float64")
    out["dest_empty_before"] = ((out["oldbalanceDest"] == 0) & (out["newbalanceDest"] == 0)).astype("float64")
    return out[FEATURES]


def state_before(df, start_step):
    """Account state as of `start_step`, only for accounts that appear at/after it (what the online store needs)."""
    hist, future = df[df["step"] < start_step], df[df["step"] >= start_step]
    state = {}
    for role, col in (("o", "nameOrig"), ("d", "nameDest")):
        need = set(future[col])
        h = hist[hist[col].isin(need)]
        for acct, g in h.groupby(col, sort=False):
            state.setdefault(acct, {})[role] = {"n": int(len(g)), "sum": float(g["amount"].sum()),
                                                 "recent": [int(s) for s in g["step"].iloc[-RECENT_CAP:]]}
    return state
