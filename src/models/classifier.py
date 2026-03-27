import os
import json
import threading
import torch
import numpy as np

from src.models.image_infer import ImageClassifier


class ClassifierService:
    """
    Singleton service wrapping ImageClassifier.
    Loaded once at app startup from environment variables.
    Thread-safe: a RLock guards load() and all predict methods.
    RLock (reentrant) is used over Lock in case load() is ever called from within a predict context.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.classifier: ImageClassifier | None = None
        self.class_names: list[str] = []
        self.model_path: str | None = None
        self.model_type: str | None = None
        self.num_classes: int = 27
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        model_path = os.environ.get("BEST_MODEL_PATH")
        model_type = os.environ.get("BEST_MODEL_TYPE", "resnet50")
        num_classes = int(os.environ.get("NUM_CLASSES", 27))
        classes_path = os.environ.get("CLASSES_JSON_PATH", "classes.json")
        # print(f"[ClassifierService] Loading: {model_type} from {model_path} on {self.device}")

        if not model_path:
            raise RuntimeError("BEST_MODEL_PATH environment variable not set")
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model file not found: {model_path}")

        if os.path.exists(classes_path):
            with open(classes_path, "r") as f:
                class_names = json.load(f)
        else:
            class_names = [str(i) for i in range(num_classes)]

        new_classifier = ImageClassifier(
            model_type=model_type,
            model_file_path=model_path,
            num_classes=num_classes,
            device=self.device,
        )

        # Swap atomically under lock so predict calls never see a half-loaded state
        with self._lock:
            self.classifier = new_classifier
            self.class_names = class_names
            self.model_path = model_path
            self.model_type = model_type
            self.num_classes = num_classes

        print(f"[ClassifierService] Model loaded: {model_type} from {model_path} on {self.device}")

    def is_loaded(self) -> bool:
        return getattr(self, 'classifier', None) is not None

    def predict_single(self, img, top_k: int = 5) -> dict:
        with self._lock:
            probs: np.ndarray = self.classifier.predict_image_mem(img, device=self.device)
            class_names = self.class_names

        class_id = int(probs.argmax())
        confidence = float(probs[class_id])
        class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)

        top_indices = probs.argsort()[::-1][:top_k]
        top_k_results = [
            {
                "class_id": int(i),
                "class_name": class_names[i] if i < len(class_names) else str(i),
                "confidence": float(probs[i]),
            }
            for i in top_indices
        ]

        return {"class_name": class_name, "class_id": class_id, "confidence": confidence, "top_k": top_k_results}

    def predict_batch(self, image_paths: list[str]) -> list[dict]:
        with self._lock:
            probs_array: np.ndarray = self.classifier.predict_batch(image_paths)
            class_names = self.class_names

        results = []
        for probs in probs_array:
            class_id = int(probs.argmax())
            confidence = float(probs[class_id])
            class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
            results.append({"class_name": class_name, "class_id": class_id, "confidence": confidence})
        return results


# Singleton
classifier_service = ClassifierService()
