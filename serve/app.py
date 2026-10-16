"""REST API over the TriageEngine.

    uvicorn serve.app:app --port 8080
"""
from __future__ import annotations

import time

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from geoint.engine import TriageEngine, decode_chip

app = FastAPI(title="geoint-triage", version="1.0.0")
engine = TriageEngine()
METRICS = {"requests": 0, "chips": 0, "routed_analyst": 0, "routed_novel": 0, "latency_ms_sum": 0.0}


def _read(files: list[UploadFile]):
    try:
        return np.stack([decode_chip(f.file.read()) for f in files])
    except Exception as e:  # bad image bytes
        raise HTTPException(400, f"could not decode image: {e}") from e


@app.get("/health")
def health():
    return {"status": "ok", "classes": engine.classes, "policy": engine.policy}


@app.post("/classify")
def classify(files: list[UploadFile] = File(...)):
    if len(files) > 512:
        raise HTTPException(413, "max 512 chips per request")
    t0 = time.perf_counter()
    rows = engine.classify(_read(files), [f.filename or str(i) for i, f in enumerate(files)])
    METRICS["requests"] += 1
    METRICS["chips"] += len(rows)
    METRICS["routed_analyst"] += sum(r["route"] == "analyst" for r in rows)
    METRICS["routed_novel"] += sum(r["reason"] == "novel" for r in rows)
    METRICS["latency_ms_sum"] += (time.perf_counter() - t0) * 1000
    return {"results": rows}


@app.post("/drift")
def drift(files: list[UploadFile] = File(...)):
    if len(files) < 30:
        raise HTTPException(400, "send at least 30 chips for a drift test")
    return engine.drift(_read(files))


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    lines = [f"geoint_{k}_total {v}" for k, v in METRICS.items()]
    return "\n".join(lines) + "\n"
