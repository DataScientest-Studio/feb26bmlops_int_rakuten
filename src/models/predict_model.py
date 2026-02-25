import json
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _resolve_model_dir(run_id: str, base_dir: str = "models/text") -> str:
	run_dir = os.path.join(base_dir, run_id)
	best_model_dir = os.path.join(run_dir, "best_model")
	if os.path.isdir(best_model_dir):
		return best_model_dir
	if os.path.isdir(run_dir):
		return run_dir
	raise FileNotFoundError(f"Run directory not found for run_id='{run_id}'")


def _load_label_map(run_id: str, base_dir: str = "models/text") -> dict:
	label_map_path = os.path.join(base_dir, run_id, "label_map.json")
	if not os.path.exists(label_map_path):
		return {}
	with open(label_map_path, "r", encoding="utf-8") as label_map_file:
		payload = json.load(label_map_file)
	return payload.get("id2label", {})


def predict_text(cfg: dict) -> dict:
	run_id = cfg.get("run_id")
	text = cfg.get("text")
	base_dir = cfg.get("base_dir", "models/text")
	max_length = cfg.get("max_length", 256)

	if not run_id:
		raise ValueError("'run_id' is required")
	if text is None or str(text).strip() == "":
		raise ValueError("'text' must be a non-empty string")

	model_dir = _resolve_model_dir(run_id=run_id, base_dir=base_dir)
	tokenizer = AutoTokenizer.from_pretrained(model_dir)
	model = AutoModelForSequenceClassification.from_pretrained(model_dir)
	model.eval()

	encoded = tokenizer(
		str(text),
		truncation=True,
		max_length=max_length,
		return_tensors="pt",
	)

	with torch.no_grad():
		logits = model(**encoded).logits
		probabilities = torch.softmax(logits, dim=-1)[0]

	predicted_id = int(torch.argmax(probabilities).item())
	confidence = float(probabilities[predicted_id].item())
	id_to_label = _load_label_map(run_id=run_id, base_dir=base_dir)
	predicted_label = id_to_label.get(str(predicted_id), str(predicted_id))

	return {
		"run_id": run_id,
		"predicted_class_id": predicted_id,
		"predicted_label": predicted_label,
		"confidence": confidence,
	}
