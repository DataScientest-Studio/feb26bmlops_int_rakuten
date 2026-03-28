import os
import re
import shutil
from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    TrainRequest,
    JobStatusResponse,
    TrainImageSyncRequest,
    TrainImageSyncResponse,
    ImageDbResetRequest,
    ImageDbUpdateRequest,
    ImageDbResponse,
)
from src.api.job_store import job_store
from src.models.trainer import start_training, run_training_sync
from src.models.mlflow_utils import log_image_training_run, evaluate_and_promote
from src.data.create_image_db import update_image_db_for_step

router = APIRouter(prefix="/train", tags=["train"])

EXPERIMENTS_DIR = os.environ.get("EXPERIMENTS_DIR", "models")


def _count_files(folder_path: str) -> int:
    total = 0
    for _, _, files in os.walk(folder_path):
        total += len(files)
    return total


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


@router.post("/sync", response_model=TrainImageSyncResponse)
async def train_sync(request: TrainImageSyncRequest):
    """
    Run image training synchronously.
    Intended for Airflow so the task only succeeds when training is finished.
    """
    try:
        resume_path = request.resume
        if request.use_transfer_learning:
            if not resume_path:
                best_model_path = os.environ.get("BEST_MODEL_PATH")
                if best_model_path and os.path.exists(best_model_path):
                    resume_path = best_model_path
            if not resume_path:
                raise ValueError(
                    "Transfer learning requested, but no resume checkpoint found. "
                    "Provide 'resume' or ensure BEST_MODEL_PATH exists."
                )
        else:
            resume_path = None

        request_payload = request.model_dump(exclude={"use_transfer_learning", "step"})
        request_payload["resume"] = resume_path
        effective_request = TrainRequest(**request_payload)

        session_name, session_folder = _get_session_info(
            model_name=effective_request.model_type.value,
            resume_path=effective_request.resume,
        )

        train_output = run_training_sync(
            request=effective_request,
            session_folder=session_folder,
            session_name=session_name,
        )

        try:
            image_metrics = log_image_training_run(
                model=train_output["model"],
                model_name=effective_request.model_type.value,
                session_folder=session_folder,
                csv_log=train_output["csv_log"],
                final_model_path=train_output["final_model_path"],
                use_transfer_learning=request.use_transfer_learning,
                resume_path=resume_path,
                step=request.step,
            )

            if image_metrics and "eval_f1_macro" in image_metrics:
                evaluate_and_promote(
                    new_metrics=image_metrics,
                    model_name=f"IMAGE_{effective_request.model_type.value.upper()}",
                )
        except Exception as log_exc:
            # Training already succeeded; do not fail API because logging failed.
            print(f"[MLflow][Image] Logging skipped due to error: {log_exc}")

        return TrainImageSyncResponse(
            status="done",
            session_name=session_name,
            session_folder=session_folder,
            final_model_path=train_output["final_model_path"],
            resume_used=resume_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/image-db/reset", response_model=ImageDbResponse)
async def reset_image_db(request: ImageDbResetRequest):
    """
    Deletes image_db folder and recreates empty train/val roots.
    """
    try:
        if os.path.exists(request.output_folder):
            shutil.rmtree(request.output_folder)

        os.makedirs(os.path.join(request.output_folder, "train"), exist_ok=True)
        os.makedirs(os.path.join(request.output_folder, "val"), exist_ok=True)

        train_dir = os.path.join(request.output_folder, "train")
        val_dir = os.path.join(request.output_folder, "val")

        return ImageDbResponse(
            status="done",
            output_folder=request.output_folder,
            step=None,
            train_file_count=_count_files(train_dir),
            val_file_count=_count_files(val_dir),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/image-db/update", response_model=ImageDbResponse)
async def update_image_db(request: ImageDbUpdateRequest):
    """
    Adds image files for a specific step into image_db/train and image_db/val.
    """
    try:
        update_image_db_for_step(
            db_url=request.db_url,
            step=request.step,
            image_column=request.image_column,
            label_column=request.label_column,
            sample=request.sample_number,
            input_folder=request.input_folder,
            output_folder=request.output_folder,
        )

        train_dir = os.path.join(request.output_folder, "train")
        val_dir = os.path.join(request.output_folder, "val")

        return ImageDbResponse(
            status="done",
            output_folder=request.output_folder,
            step=request.step,
            train_file_count=_count_files(train_dir),
            val_file_count=_count_files(val_dir),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
