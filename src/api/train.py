import os
import re
from fastapi import APIRouter

from src.api.schemas import TrainRequest, JobStatusResponse
from src.api.job_store import job_store
from src.models.trainer import start_training

router = APIRouter(prefix="/train", tags=["train"])

EXPERIMENTS_DIR = os.environ.get("EXPERIMENTS_DIR", "models")


def _get_session_info(model_name: str, resume_path: str = None):
    """Mirrors get_session_info from Train_Main.py"""
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    pattern = re.compile(rf"{model_name}_(\d+)")
    existing_ids = []
    for d in os.listdir(EXPERIMENTS_DIR):
        match = pattern.match(d)
        if match:
            existing_ids.append(int(match.group(1)))

    session_id = max(existing_ids) + 1 if existing_ids else 1
    session_name = f"{model_name}_{session_id:02d}"

    if resume_path:
        parent_match = re.search(rf"({model_name}_\d+_epoch_\d+)", resume_path)
        if parent_match:
            session_name += f"_from_{parent_match.group(1)}"

    session_folder = os.path.join(EXPERIMENTS_DIR, session_name)
    os.makedirs(session_folder, exist_ok=True)

    return session_name, session_folder


@router.post("", response_model=JobStatusResponse, status_code=202)
async def start_train(request: TrainRequest):
    """
    Start a training job asynchronously.
    Returns a job_id to poll for progress via GET /jobs/{job_id}.
    """
    session_name, session_folder = _get_session_info(
        model_name=request.model_type.value,
        resume_path=request.resume,
    )

    job = job_store.create_job(
        total_epochs=request.epochs,
        session_folder=session_folder,
    )

    start_training(
        job_id=job.job_id,
        request=request,
        session_folder=session_folder,
        session_name=session_name,
    )

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        total_epochs=job.total_epochs,
        session_folder=session_folder,
    )
