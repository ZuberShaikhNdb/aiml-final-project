import pandas as pd
from .features import SCORED_TYPES

PAYSIM_COLS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
               "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"]


def load_paysim(path):
    """Read PaySim, keep scored types, rename to the names used everywhere, keep ARRIVAL ORDER."""
    df = pd.read_csv(path, usecols=PAYSIM_COLS)
    df = df[df["type"].isin(SCORED_TYPES)].copy()
    df = df.rename(columns={"newbalanceOrig": "newbalanceOrg"})
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    return df


def split_masks(df, train_end, val_end):
    s = df["step"]
    return s <= train_end, (s > train_end) & (s <= val_end), s > val_end
