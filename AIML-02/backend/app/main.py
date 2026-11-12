import re, threading
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from . import config, db, orthanc, pipeline, poller

stop = threading.Event()


@asynccontextmanager
async def lifespan(app):
    db.conn()
    if not config.DISABLE_POLLER:
        threading.Thread(target=poller.run, args=(stop,), daemon=True).start()
    yield
    stop.set()


app = FastAPI(title="AI-Assisted Medical Imaging API", version="1.0.0", lifespan=lifespan,
              description=config.DISCLAIMER)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
ID_RE = re.compile(r"^[0-9a-f-]{10,64}$")


class Review(BaseModel):
    decision: Literal["approved", "rejected"]
    reviewer: str = Field(min_length=1, max_length=80)
    comment: str = Field(default="", max_length=1000)


@app.get("/health")
def health():
    return {"status": "ok", "demo_mode": config.DEMO_MODE}


@app.get("/config")
def get_config():
    return {"demo_mode": config.DEMO_MODE, "disclaimer": config.DISCLAIMER}


@app.get("/studies")
def studies():
    return db.list_studies()


@app.get("/studies/{oid}")
def study(oid: str):
    s = db.get_study(oid)
    if not s:
        raise HTTPException(404, "Study not found")
    return s


@app.get("/studies/{oid}/image/{kind}")
def image(oid: str, kind: Literal["preview", "overlay"]):
    p = config.DATA_DIR / "images" / f"{oid}_{kind}.png"
    if not ID_RE.match(oid) or not p.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(p, media_type="image/png")


@app.post("/studies/{oid}/review")
def review(oid: str, body: Review):
    row = db.add_review(oid, body.reviewer.strip(), body.decision, body.comment.strip())
    if row is None:
        raise HTTPException(404, "Study not found or has no AI result to review")
    return row


@app.post("/studies/{oid}/reprocess")
def reprocess(oid: str):
    pipeline.process_study(oid)
    return db.get_study(oid)


@app.get("/audit")
def audit():
    return db.list_audit()


@app.get("/audit/verify")
def audit_verify():
    return db.verify_audit()


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    sent = 0
    for f in files:
        try:
            orthanc.upload_instance(await f.read())
            sent += 1
        except Exception as e:
            raise HTTPException(502, f"Orthanc rejected {f.filename}: {e}")
    return {"uploaded": sent, "note": "Analysis starts once Orthanc marks the study stable (~5 s)."}
