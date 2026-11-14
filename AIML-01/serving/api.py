"""FastAPI scoring service. Fetches account features, scores with XGBoost, logs the event, returns a fraud score."""
import os, sys
from pathlib import Path
from typing import Optional
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fraud.alerts import STATUSES, EventLog
from fraud.rings import find_rings
from fraud.scoring import Scorer
from fraud.store import make_store

MODEL_DIR = Path(os.getenv("MODEL_DIR", "weights"))
TAG = os.getenv("MODEL_TAG", "full")
app = FastAPI(title="Real-Time Fraud Scoring API", version="1.0.0",
              description="Educational project. Scores PaySim-style transactions; not a production fraud system.")
_state = {}


def get_scorer():
    if "scorer" not in _state:
        store = make_store(os.getenv("STORE", "memory"), MODEL_DIR / "online_state.pkl" if os.getenv("STORE", "memory") == "memory"
                           and (MODEL_DIR / "online_state.pkl").exists() else None, os.getenv("FEAST_REPO", "feature_repo"))
        log = EventLog(os.getenv("DB_PATH", "events.db"))
        _state["scorer"] = Scorer(MODEL_DIR / f"xgb_{TAG}.json", MODEL_DIR / f"xgb_{TAG}_meta.json", store, log)
    return _state["scorer"]


class Transaction(BaseModel):
    step: int = Field(ge=0)
    type: str
    amount: float = Field(ge=0)
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrg: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float
    true_label: Optional[int] = Field(default=None, description="Ground truth, only present in replay/demo data; never used for scoring")


class Case(BaseModel):
    status: str
    label: Optional[int] = Field(default=None, ge=0, le=1, description="1 = fraud, 0 = legitimate")
    analyst: str = ""
    note: str = ""


@app.get("/health")
def health():
    s = get_scorer()
    return {"status": "ok", "model": TAG, "threshold": s.threshold, "balance_features": s.meta["balance_features"]}


@app.post("/score")
def score(tx: Transaction):
    try:
        return get_scorer().score(tx.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/alerts")
def alerts(status: Optional[str] = None, limit: int = 200):
    return get_scorer().log.alerts(status, limit)


@app.post("/alerts/{alert_id}/case")
def case(alert_id: int, body: Case):
    if body.status not in STATUSES:
        raise HTTPException(422, f"status must be one of {STATUSES}")
    if not get_scorer().log.set_case(alert_id, body.status, body.label, body.analyst, body.note):
        raise HTTPException(404, "Alert not found")
    return {"ok": True}


@app.get("/rings")
def rings():
    return find_rings(get_scorer().log.alerts(limit=100000))


@app.get("/stats")
def stats():
    return get_scorer().log.stats()


@app.get("/feedback")
def feedback():
    return get_scorer().log.feedback()
