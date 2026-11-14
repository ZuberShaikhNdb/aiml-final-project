import json, time
import numpy as np
import pandas as pd
import xgboost as xgb
from .features import SCORED_TYPES, online_features, update_state

REQUIRED = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrg", "nameDest", "oldbalanceDest", "newbalanceDest"]


class Scorer:
    def __init__(self, model_path, meta_path, store, log):
        self.meta = json.load(open(meta_path))
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.features = self.meta["features"]
        self.threshold = self.meta["threshold"]
        self.store, self.log = store, log

    def score(self, tx):
        t0 = time.perf_counter()
        missing = [k for k in REQUIRED if k not in tx]
        if missing:
            raise ValueError(f"missing fields: {missing}")
        if tx["type"] not in SCORED_TYPES:  # PaySim fraud only occurs in TRANSFER / CASH_OUT
            res = {"fraud_score": 0.0, "decision": "not_scored", "reasons": [f"type {tx['type']} is not scored"], "threshold": self.threshold}
            res["latency_ms"] = (time.perf_counter() - t0) * 1000
            return res
        o = self.store.get(tx["nameOrig"]).get("o")
        d = self.store.get(tx["nameDest"]).get("d")
        feats = online_features(tx, o, d)
        X = pd.DataFrame([feats])[self.features]
        p = float(self.model.predict_proba(X)[0, 1])
        contrib = self.model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
        top = np.argsort(-np.abs(contrib))[:3]
        reasons = [f"{self.features[i]}={X.iloc[0, i]:.4g} ({'raises' if contrib[i] > 0 else 'lowers'} risk)" for i in top]
        self.store.put(tx["nameOrig"], "o", update_state(o, tx["amount"], tx["step"]))
        self.store.put(tx["nameDest"], "d", update_state(d, tx["amount"], tx["step"]))
        res = {"fraud_score": p, "decision": "alert" if p >= self.threshold else "pass", "reasons": reasons, "threshold": self.threshold}
        res["latency_ms"] = (time.perf_counter() - t0) * 1000
        res["event_id"] = self.log.add(tx, res, feats)
        return res
