from fastapi import APIRouter, HTTPException

from src.api.job_store import job_store
from src.api.schemas import JobStatusResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _job_to_response(job) -> JobStatusResponse:
    return JobStatusResponse(**job.__dict__)

@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status and latest metrics for a training job."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _job_to_response(job)


@router.get("", response_model=dict[str, JobStatusResponse])
async def list_jobs():
    """List all training jobs."""
    return {job_id: _job_to_response(job) for job_id, job in job_store.list_jobs().items()}
