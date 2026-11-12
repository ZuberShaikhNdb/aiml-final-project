"""SQLite storage: study results + an append-only, hash-chained audit log."""
import hashlib, json, sqlite3, threading, time
from . import config

_lock = threading.RLock()
_conn = None
SCHEMA = """
CREATE TABLE IF NOT EXISTS studies(
  orthanc_id TEXT PRIMARY KEY, study_uid TEXT, modality TEXT, kind TEXT, top_label TEXT,
  confidence REAL, details TEXT, model_version TEXT, status TEXT DEFAULT 'pending',
  error TEXT, created REAL);
CREATE TABLE IF NOT EXISTS audit(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, orthanc_id TEXT, reviewer TEXT,
  decision TEXT, comment TEXT, ai_label TEXT, ai_confidence REAL, model_version TEXT,
  prev_hash TEXT, hash TEXT);
CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
"""


def conn():
    global _conn
    if _conn is None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(config.DATA_DIR / "app.db", check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(SCHEMA)
    return _conn


def _study(row):
    d = dict(row)
    d["details"] = json.loads(d["details"] or "{}")
    return d


def save_study(rec: dict):
    with _lock:
        conn().execute(
            "INSERT OR REPLACE INTO studies(orthanc_id,study_uid,modality,kind,top_label,confidence,"
            "details,model_version,status,error,created) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (rec["orthanc_id"], rec.get("study_uid"), rec.get("modality"), rec.get("kind"),
             rec.get("top_label"), rec.get("confidence"), json.dumps(rec.get("details", {})),
             rec.get("model_version"), rec.get("status", "pending"), rec.get("error"),
             rec.get("created", time.time())))
        conn().commit()


def list_studies():
    with _lock:
        return [_study(r) for r in conn().execute("SELECT * FROM studies ORDER BY created DESC")]


def get_study(oid):
    with _lock:
        r = conn().execute("SELECT * FROM studies WHERE orthanc_id=?", (oid,)).fetchone()
        return _study(r) if r else None


def _digest(prev, fields):
    return hashlib.sha256((prev + json.dumps(fields)).encode()).hexdigest()


def add_review(oid, reviewer, decision, comment):
    with _lock:
        s = get_study(oid)
        if s is None or s["status"] == "error":
            return None
        last = conn().execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
        prev = last["hash"] if last else "GENESIS"
        ts = round(time.time(), 3)
        fields = [ts, oid, reviewer, decision, comment, s["top_label"], s["confidence"], s["model_version"]]
        h = _digest(prev, fields)
        conn().execute(
            "INSERT INTO audit(ts,orthanc_id,reviewer,decision,comment,ai_label,ai_confidence,"
            "model_version,prev_hash,hash) VALUES(?,?,?,?,?,?,?,?,?,?)", (*fields, prev, h))
        conn().execute("UPDATE studies SET status=? WHERE orthanc_id=?", (decision, oid))
        conn().commit()
        return dict(conn().execute("SELECT * FROM audit ORDER BY id DESC LIMIT 1").fetchone())


def list_audit():
    with _lock:
        return [dict(r) for r in conn().execute("SELECT * FROM audit ORDER BY id DESC")]


def verify_audit():
    """Recompute the hash chain. Any edited or deleted row breaks it."""
    with _lock:
        prev, n = "GENESIS", 0
        for r in conn().execute("SELECT * FROM audit ORDER BY id"):
            fields = [r["ts"], r["orthanc_id"], r["reviewer"], r["decision"], r["comment"],
                      r["ai_label"], r["ai_confidence"], r["model_version"]]
            if r["prev_hash"] != prev or r["hash"] != _digest(prev, fields):
                return {"ok": False, "checked": n, "broken_at": r["id"]}
            prev, n = r["hash"], n + 1
        return {"ok": True, "checked": n, "broken_at": None}


def kv_get(k, default=None):
    with _lock:
        r = conn().execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return r["v"] if r else default


def kv_set(k, v):
    with _lock:
        conn().execute("INSERT OR REPLACE INTO kv(k,v) VALUES(?,?)", (k, v))
        conn().commit()
