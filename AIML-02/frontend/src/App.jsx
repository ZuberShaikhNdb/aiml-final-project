import { useCallback, useEffect, useState } from "react";

async function api(path, opts) {
  const r = await fetch(`/api${path}`, opts);
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : r.statusText);
  }
  return r.json();
}
const pct = (x) => `${(x * 100).toFixed(1)}%`;
const when = (t) => new Date(t * 1000).toLocaleString();

export default function App() {
  const [cfg, setCfg] = useState({ demo_mode: false, disclaimer: "" });
  const [studies, setStudies] = useState([]);
  const [selId, setSelId] = useState(null);
  const [view, setView] = useState("review");
  const [audit, setAudit] = useState([]);
  const [integrity, setIntegrity] = useState(null);
  const [reviewer, setReviewer] = useState(localStorage.getItem("reviewer") || "");
  const [comment, setComment] = useState("");
  const [overlay, setOverlay] = useState(0.8);
  const [msg, setMsg] = useState("");

  const refresh = useCallback(async () => {
    try {
      setStudies(await api("/studies"));
      if (view === "audit") setAudit(await api("/audit"));
    } catch (e) { setMsg(`Cannot reach the API: ${e.message}`); }
  }, [view]);

  useEffect(() => { api("/config").then(setCfg).catch(() => {}); }, []);
  useEffect(() => { refresh(); const t = setInterval(refresh, 5000); return () => clearInterval(t); }, [refresh]);
  useEffect(() => { localStorage.setItem("reviewer", reviewer); }, [reviewer]);

  const sel = studies.find((s) => s.orthanc_id === selId) || studies[0];

  async function decide(decision) {
    setMsg("");
    try {
      await api(`/studies/${sel.orthanc_id}/review`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decision, reviewer, comment }),
      });
      setComment(""); setMsg(`Recorded: ${decision}`); refresh();
    } catch (e) { setMsg(e.message); }
  }

  async function upload(e) {
    const fd = new FormData();
    [...e.target.files].forEach((f) => fd.append("files", f));
    setMsg("Uploading…");
    try { setMsg((await api("/upload", { method: "POST", body: fd })).note); } catch (err) { setMsg(err.message); }
    e.target.value = "";
  }

  return (
    <div className="app">
      <header>
        <h1>Imaging review</h1>
        <nav>
          <button className={view === "review" ? "on" : ""} onClick={() => setView("review")}>Studies</button>
          <button className={view === "audit" ? "on" : ""} onClick={() => setView("audit")}>Audit log</button>
        </nav>
        <label className="file">Upload DICOM<input type="file" multiple onChange={upload} /></label>
      </header>

      <div className="notice" role="note">{cfg.disclaimer || "For education only. Not for clinical use."}</div>
      {cfg.demo_mode && <div className="notice demo">Demo mode: results come from an untrained placeholder, not a real model.</div>}
      {msg && <div className="msg" role="status">{msg}</div>}

      {view === "audit" ? (
        <section className="audit">
          <div className="row">
            <button onClick={async () => setIntegrity(await api("/audit/verify"))}>Verify log integrity</button>
            {integrity && <span className={integrity.ok ? "good" : "bad"}>
              {integrity.ok ? `Chain intact (${integrity.checked} entries)` : `Chain broken at entry ${integrity.broken_at}`}</span>}
          </div>
          {audit.length === 0 ? <p className="empty">No decisions recorded yet. Approve or reject a study to create the first entry.</p> : (
            <table>
              <thead><tr><th>Time</th><th>Study</th><th>Reviewer</th><th>Decision</th><th>AI result</th><th>Comment</th></tr></thead>
              <tbody>{audit.map((a) => (
                <tr key={a.id}><td>{when(a.ts)}</td><td title={a.orthanc_id}>{a.orthanc_id.slice(0, 8)}</td><td>{a.reviewer}</td>
                  <td className={a.decision}>{a.decision}</td><td>{a.ai_label} ({pct(a.ai_confidence || 0)})</td><td>{a.comment}</td></tr>))}</tbody>
            </table>)}
        </section>
      ) : (
        <main>
          <aside className="list">
            {studies.length === 0 && <p className="empty">No studies yet. Send DICOM to Orthanc (port 4242) or use Upload DICOM.</p>}
            {studies.map((s) => (
              <button key={s.orthanc_id} className={`study ${sel?.orthanc_id === s.orthanc_id ? "on" : ""}`} onClick={() => setSelId(s.orthanc_id)}>
                <b>{s.status === "error" ? "Failed" : s.top_label}</b>
                <span>{s.modality || "—"} · {s.orthanc_id.slice(0, 8)}</span>
                <em className={s.status}>{s.status}</em>
              </button>))}
          </aside>

          {sel && sel.status !== "error" && (
            <>
              <section className="viewer">
                <div className="frame">
                  <img alt="Scan" src={`/api/studies/${sel.orthanc_id}/image/preview?v=${sel.created}`} />
                  <img alt={sel.kind === "segmentation" ? "Segmentation overlay" : "Grad-CAM heatmap"} className="ov"
                       style={{ opacity: overlay }} src={`/api/studies/${sel.orthanc_id}/image/overlay?v=${sel.created}`} />
                </div>
                <label className="slider">AI overlay
                  <input type="range" min="0" max="1" step="0.05" value={overlay} onChange={(e) => setOverlay(+e.target.value)} />
                </label>
              </section>

              <section className="panel">
                <h2>{sel.top_label}</h2>
                <div className="conf" aria-label={`Confidence ${pct(sel.confidence)}`}>
                  <div style={{ width: pct(sel.confidence) }} /><span>{pct(sel.confidence)} confidence</span>
                </div>
                <p className="small">Model {sel.model_version} · {sel.kind}</p>
                {sel.kind === "classification" ? (
                  <ul className="probs">{Object.entries(sel.details.predictions).slice(0, 6).map(([k, v]) => (
                    <li key={k}><span>{k.replace("_", " ")}</span><i><u style={{ width: pct(v) }} /></i><b>{pct(v)}</b></li>))}</ul>
                ) : (
                  <p className="small">Segmented volume {sel.details.volume_ml} mL · shown slice {sel.details.slice_index + 1} of {sel.details.n_slices}</p>
                )}
                <label>Reviewer<input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="Your name" /></label>
                <label>Comment<textarea value={comment} onChange={(e) => setComment(e.target.value)} rows={3} /></label>
                <div className="actions">
                  <button className="approve" disabled={!reviewer.trim()} onClick={() => decide("approved")}>Approve</button>
                  <button className="reject" disabled={!reviewer.trim()} onClick={() => decide("rejected")}>Reject</button>
                </div>
                <p className="small">Status: <em className={sel.status}>{sel.status}</em>. Every decision is added to the audit log.</p>
              </section>
            </>
          )}
          {sel && sel.status === "error" && <section className="panel err"><h2>Analysis failed</h2><p>{sel.error}</p></section>}
        </main>
      )}
    </div>
  );
}
