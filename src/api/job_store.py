import threading
from .schemas import JobStatus
import uuid


class Job:
    def __init__(self, job_id: str, total_epochs: int, session_folder: str):
        self.job_id = job_id
        self.status = JobStatus.pending
        self.current_epoch = 0
        self.total_epochs = total_epochs
        self.last_train_loss: float | None
        self.last_val_loss: float | None
        self.last_val_accuracy: float | None
        self.last_val_f1: float | None
        self.session_folder = session_folder
        self.error: str | None


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create_job(self, total_epochs: int, session_folder: str) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(job_id=job_id, total_epochs=total_epochs, session_folder=session_folder)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for k, v in kwargs.items():
                    setattr(job, k, v)

    def list_jobs(self) -> dict[str, Job]:
        with self._lock:
            return dict(self._jobs)


# Singleton
job_store = JobStore()
