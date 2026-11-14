"""Analyst dashboard (Streamlit): alerts, case status, feedback labels, linked fraud rings."""
import json
import os
import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
STATUSES = ["new", "investigating", "confirmed_fraud", "false_positive"]
st.set_page_config(page_title="Fraud analyst dashboard", layout="wide")
st.title("Fraud analyst dashboard")
st.caption("Educational project on simulated PaySim transactions. Not for real fraud decisions.")


def get(path, **params):
    r = requests.get(f"{API}{path}", params=params, timeout=15)
    r.raise_for_status()
    return r.json()


try:
    s = get("/stats")
except Exception as e:
    st.error(f"Cannot reach the scoring API at {API}: {e}")
    st.stop()

c = st.columns(5)
c[0].metric("Scored", f"{int(s['n']):,}")
c[1].metric("Alerts", f"{int(s['alerts']):,}")
c[2].metric("Confirmed fraud", int(s["confirmed"]))
c[3].metric("False positives", int(s["false_pos"]))
c[4].metric("Latency avg / p95", f"{s['avg_ms']:.1f} / {s['p95_ms']:.1f} ms")
if s["true_fraud"]:
    st.caption(f"Replay ground truth: {int(s['caught'])} of {int(s['true_fraud'])} true frauds alerted so far.")

tab1, tab2, tab3 = st.tabs(["Alerts", "Fraud rings", "Feedback labels"])
with tab1:
    flt = st.selectbox("Case status", ["all"] + STATUSES)
    rows = get("/alerts", **({} if flt == "all" else {"status": flt}), limit=300)
    if not rows:
        st.info("No alerts yet. Start the replay producer to stream transactions.")
    else:
        df = pd.DataFrame(rows)[["id", "step", "type", "nameOrig", "nameDest", "amount", "score", "status", "reasons"]]
        st.dataframe(df, hide_index=True)
        st.subheader("Review a case")
        pick = st.selectbox("Alert id", df["id"])
        row = next(r for r in rows if r["id"] == pick)
        st.write("Why it was flagged:", ", ".join(json.loads(row["reasons"])))
        with st.form("case"):
            status = st.selectbox("New status", STATUSES, index=STATUSES.index(row["status"]))
            label = st.radio("Verdict (feedback label)", ["not set", "fraud", "legitimate"], horizontal=True)
            analyst, note = st.text_input("Analyst"), st.text_area("Note")
            if st.form_submit_button("Save"):
                lab = {"not set": None, "fraud": 1, "legitimate": 0}[label]
                requests.post(f"{API}/alerts/{pick}/case", json={"status": status, "label": lab, "analyst": analyst, "note": note}, timeout=15).raise_for_status()
                st.success("Saved. Refresh to see it in the table.")
with tab2:
    rings = get("/rings")
    st.write("Alerts that share an account are linked into one ring.")
    if not rings:
        st.info("No linked alerts yet.")
    for r in rings[:20]:
        with st.expander(f"Ring {r['ring_id']}: {r['size']} alerts, {len(r['accounts'])} accounts"):
            st.write("Alert ids:", r["alert_ids"])
            st.write("Accounts:", r["accounts"])
with tab3:
    fb = get("/feedback")
    st.write(f"{len(fb)} analyst-labelled alerts. These are the labels to merge into the next retraining run.")
    if fb:
        st.dataframe(pd.DataFrame(fb)[["id", "nameOrig", "nameDest", "amount", "score", "analyst_label", "analyst", "note"]], hide_index=True)
