"""Online account-state stores. MemoryStore (tests, local demo) and FeastStore (feature store)."""
import pickle
from datetime import datetime, timezone
import pandas as pd

ROLES = ("o", "d")


class MemoryStore:
    def __init__(self, state=None):
        self.s = state or {}

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    def get(self, account):
        return self.s.get(account, {})

    def put(self, account, role, st):
        self.s.setdefault(account, {})[role] = st


class FeastStore:
    """Reads account features from the Feast online store and writes updated state back."""
    VIEW = "account_stats"

    def __init__(self, repo_path):
        from feast import FeatureStore
        self.fs = FeatureStore(repo_path=repo_path)
        self.names = [f"{self.VIEW}:{r}_{k}" for r in ROLES for k in ("n", "sum", "recent")]

    def get(self, account):
        d = self.fs.get_online_features(features=self.names, entity_rows=[{"account_id": account}]).to_dict()
        out = {}
        for r in ROLES:
            n = d[f"{r}_n"][0]
            if n is not None:
                out[r] = {"n": int(n), "sum": float(d[f"{r}_sum"][0] or 0.0), "recent": list(d[f"{r}_recent"][0] or [])}
        return out

    def put(self, account, role, st):
        cur = self.get(account)
        cur[role] = st
        row = {"account_id": account, "event_timestamp": datetime.now(timezone.utc)}
        for r in ROLES:
            s = cur.get(r, {"n": 0, "sum": 0.0, "recent": []})
            row[f"{r}_n"], row[f"{r}_sum"], row[f"{r}_recent"] = int(s["n"]), float(s["sum"]), [int(x) for x in s["recent"]]
        self.fs.write_to_online_store(feature_view_name=self.VIEW, df=pd.DataFrame([row]))


def make_store(kind, state_path=None, feast_repo=None):
    if kind == "feast":
        return FeastStore(feast_repo)
    return MemoryStore.load(state_path) if state_path else MemoryStore()
