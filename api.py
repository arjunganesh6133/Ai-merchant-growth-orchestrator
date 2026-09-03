"""
FastAPI wrapper around the CrewAI campaign generator.

Runs the same crew as main.py, but over HTTP instead of the terminal, so
other programs (or a frontend) can request a campaign and poll for the
result — since a full crew run takes 15-60+ seconds, this returns a job ID
immediately instead of making the caller wait on one long request.

Run locally with:
    uvicorn api:app --reload

Then open http://127.0.0.1:8000/docs for interactive API docs (FastAPI
generates this automatically from the code below — nothing extra to write).
"""

import uuid
from datetime import datetime, timezone
from threading import Lock

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from llm_config import get_llm
from main import run_campaign_creator
from models import MerchantCampaignInput

app = FastAPI(
    title="Multi-Agent Marketing Campaign Generator",
    description="Kicks off a 4-agent CrewAI crew (Research, Copywriter, "
                 "Art Director, Manager) to produce a merchant marketing campaign brief.",
    version="1.0.0",
)

_jobs: dict[str, dict] = {}
_jobs_lock = Lock()


class CampaignRequest(MerchantCampaignInput):
    pass


class CampaignResponse(BaseModel):
    job_id: str
    status: str


def _execute_campaign(job_id: str, campaign: MerchantCampaignInput) -> None:
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"

    try:
        result = run_campaign_creator(campaign)
        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = str(result)
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)


@app.get("/health")
def health():
    try:
        get_llm()
        llm_ready = True
    except EnvironmentError:
        llm_ready = False
    return {"status": "ok", "llm_configured": llm_ready}


@app.post("/campaigns", response_model=CampaignResponse, status_code=202)
def create_campaign(request: CampaignRequest, background_tasks: BackgroundTasks):
    try:
        get_llm()
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "campaign": request.model_dump(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "result": None,
            "error": None,
        }

    background_tasks.add_task(_execute_campaign, job_id, request)
    return CampaignResponse(job_id=job_id, status="queued")


@app.get("/campaigns/{job_id}")
def get_campaign(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"job_id": job_id, **job}