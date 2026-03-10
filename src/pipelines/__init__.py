"""Pipeline modules for model training and inference."""

from .text_pipeline import train_text_bert_from_csv

train_text = train_text_bert_from_csv

__all__ = ["train_text_bert_from_csv", "train_text"]
