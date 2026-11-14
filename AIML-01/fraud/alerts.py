"""SQLite log of every scored transaction, with analyst case status and feedback labels."""
import json, sqlite3, threading, time

SCHEMA = """CREATE TABLE IF NOT EXISTS events(
 id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, step INTEGER, type TEXT, nameOrig TEXT, nameDest TEXT,
 amount REAL, score REAL, decision TEXT, reasons TEXT, features TEXT, latency_ms REAL, true_label INTEGER,
 status TEXT DEFAULT 'new', analyst_label INTEGER, analyst TEXT, note TEXT, updated REAL)"""
STATUSES = ("new", "investigating", "confirmed_fraud", "false_positive")


class EventLog:
    def __init__(self, path):
        self.c = sqlite3.connect(path, check_same_thread=False)
        self.c.row_factory = sqlite3.Row
        self.c.execute(SCHEMA)
        self.c.commit()
        self.lock = threading.Lock()

    def add(self, tx, res, feats):
        with self.lock:
            cur = self.c.execute(
                "INSERT INTO events(ts,step,type,nameOrig,nameDest,amount,score,decision,reasons,features,latency_ms,true_label)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), tx["step"], tx["type"], tx["nameOrig"], tx["nameDest"], tx["amount"], res["fraud_score"],
                 res["decision"], json.dumps(res.get("reasons", [])), json.dumps(feats), res["latency_ms"], tx.get("true_label")))
            self.c.commit()
            return cur.lastrowid

    def alerts(self, status=None, limit=200):
        q, args = "SELECT * FROM events WHERE decision='alert'", []
        if status:
            q, args = q + " AND status=?", [status]
        with self.lock:
            return [dict(r) for r in self.c.execute(q + " ORDER BY score DESC LIMIT ?", (*args, limit))]

    def set_case(self, alert_id, status, label=None, analyst="", note=""):
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        with self.lock:
            cur = self.c.execute("UPDATE events SET status=?, analyst_label=?, analyst=?, note=?, updated=? WHERE id=? AND decision='alert'",
                                 (status, label, analyst, note, time.time(), alert_id))
            self.c.commit()
            return cur.rowcount == 1

    def recent_features(self, n=5000):
        with self.lock:
            rows = self.c.execute("SELECT features FROM events ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [json.loads(r["features"]) for r in rows]

    def feedback(self):
        with self.lock:
            return [dict(r) for r in self.c.execute("SELECT * FROM events WHERE analyst_label IS NOT NULL")]

    def stats(self):
        with self.lock:
            r = self.c.execute("SELECT COUNT(*) n, SUM(decision='alert') alerts, SUM(status='confirmed_fraud') confirmed,"
                               " SUM(status='false_positive') false_pos, SUM(status='investigating') investigating,"
                               " SUM(true_label=1) true_fraud, SUM(decision='alert' AND true_label=1) caught,"
                               " AVG(latency_ms) avg_ms FROM events").fetchone()
            lat = [x[0] for x in self.c.execute("SELECT latency_ms FROM events ORDER BY latency_ms")]
        d = {k: (r[k] or 0) for k in r.keys()}
        d["p95_ms"] = lat[int(0.95 * (len(lat) - 1))] if lat else 0
        return d
