import os
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.api.job_store import job_store
from src.api.schemas import JobStatus, TrainRequest
from src.models.Train_Test import getImageLoader, prepare_model, get_optimizer, trainTestModel
from src.models.classifier import classifier_service

#TODO: create the two 
DATA_TRAIN = os.environ.get("DATA_TRAIN", "data/image_db/train")
DATA_TEST = os.environ.get("DATA_TEST", "data/image_db/test")


IMAGE_DIR = os.path.dirname(
    os.environ.get("BEST_MODEL_PATH", "models/image/resnet50_latest.model")
)


class Trainer:
    def __init__(self, request: TrainRequest):
        self.request = request
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.optimizer = None
        self.scheduler = None
        self.dataloader_train = None
        self.dataloader_test = None
        self.criterion = None

    def setup(self):
        """Load data, model, optimizer, scheduler, criterion."""
        os.makedirs(IMAGE_DIR, exist_ok=True)
        mapping_save_path = os.path.join(IMAGE_DIR, "classes.json")
        self.dataloader_train, self.dataloader_test = getImageLoader(
            train_path=DATA_TRAIN,
            test_path=DATA_TEST,
            save_mapping_to=mapping_save_path,
        )

        train_classes = self.dataloader_train.dataset.targets
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_classes),
            y=train_classes,
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights_tensor,
            label_smoothing=self.request.label_smoothing,
        )

        self.model, self.preprocess, _ = prepare_model(
            model_name=self.request.model_type.value,
            num_classes=int(os.environ.get("NUM_CLASSES", 27)),
            fine_tune_type=self.request.mode.value,
            checkpoint_path=self.request.resume,
            dropout=self.request.dropout,
        )
        self.model.to(self.device)

        self.optimizer = get_optimizer(
            self.model,
            self.request.model_type.value,
            self.request.mode.value,
            lr_classifier=self.request.lr_cls,
            lr_backbone=self.request.lr_back,
        )

        r = self.request
        if r.scheduler.value == "steplr":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=r.step_size, gamma=r.gamma)
        elif r.scheduler.value == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=r.gamma, patience=5)
        else:
            # self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=2, eta_min=1e-5)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)


    def train(self, on_epoch_end=None):
        """Run training. Blocks until complete."""
        model_name = self.request.model_type.value
        self.csv_log = os.path.join(IMAGE_DIR, f"{model_name}.csv")

        trainTestModel(
            model=self.model,
            epochs=self.request.epochs,
            dataloader_train=self.dataloader_train,
            dataloader_test=self.dataloader_test,
            preprocess=self.preprocess,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            log_file=self.csv_log,
            device=self.device,
            criterion=self.criterion,
            cm_every=self.request.cm_every,
            on_epoch_end=on_epoch_end,
        )

        final_path = os.environ.get(
            "BEST_MODEL_PATH",
            os.path.join(IMAGE_DIR, f"{model_name}_latest.model"),
        )
        os.makedirs(os.path.dirname(os.path.abspath(final_path)), exist_ok=True)
        torch.save(self.model.state_dict(), final_path)
        print(f"[Trainer] Model saved to {final_path}")

        classifier_service.load()
        print("[Trainer] Classifier reloaded with new model")

        return final_path


def _run_training(job_id: str, request: TrainRequest):
    job_store.update_job(job_id, status=JobStatus.running)

    def on_epoch_end(epoch, train_loss, val_loss, val_acc, val_f1):
        job_store.update_job(
            job_id,
            current_epoch=epoch,
            last_train_loss=train_loss,
            last_val_loss=val_loss,
            last_val_accuracy=val_acc,
            last_val_f1=val_f1,
        )

    try:
        trainer = Trainer(request)
        trainer.setup()
        trainer.train(on_epoch_end=on_epoch_end)
        job_store.update_job(job_id, status=JobStatus.done)
    except Exception as e:
        job_store.update_job(job_id, status=JobStatus.failed, error=str(e))
        raise


def start_training(job_id: str, request: TrainRequest):
    thread = threading.Thread(
        target=_run_training,
        args=(job_id, request),
        daemon=True,
    )
    thread.start()


def run_training_sync(request: TrainRequest) -> dict:
    """Run image training synchronously and return output details."""
    trainer = Trainer(request)
    trainer.setup()
    final_model_path = trainer.train()
    return {
        "final_model_path": final_model_path,
        "csv_log": trainer.csv_log,
        "model": trainer.model,
    }
