import sqlite3
from fastapi.testclient import TestClient
from app import config, db, pipeline, sample
from app.main import app

client = TestClient(app)
OID = "abcdef1234567890"


def _seed():
    pipeline.process_blobs(OID, [sample.make_xray(3)])


def test_health_and_disclaimer():
    assert client.get("/health").json()["demo_mode"] is True
    assert "not for clinical use" in client.get("/config").json()["disclaimer"].lower()


def test_review_flow_and_audit_chain():
    _seed()
    assert client.get(f"/studies/{OID}").json()["status"] == "pending"
    assert client.get(f"/studies/{OID}/image/overlay").status_code == 200
    r = client.post(f"/studies/{OID}/review", json={"decision": "approved", "reviewer": "Dr. A", "comment": "ok"})
    assert r.status_code == 200
    client.post(f"/studies/{OID}/review", json={"decision": "rejected", "reviewer": "Dr. B"})
    assert client.get(f"/studies/{OID}").json()["status"] == "rejected"
    assert len(client.get("/audit").json()) >= 2
    assert client.get("/audit/verify").json()["ok"] is True


def test_validation_and_unknown_study():
    assert client.post(f"/studies/{OID}/review", json={"decision": "maybe", "reviewer": "x"}).status_code == 422
    assert client.post("/studies/deadbeef00/review", json={"decision": "approved", "reviewer": "x"}).status_code == 404
    assert client.get("/studies/../../etc/passwd/image/preview").status_code == 404


def test_tampering_breaks_chain():
    _seed()
    client.post(f"/studies/{OID}/review", json={"decision": "approved", "reviewer": "Dr. C"})
    db.conn().execute("UPDATE audit SET decision='rejected' WHERE id=1")
    db.conn().commit()
    assert client.get("/audit/verify").json()["ok"] is False
