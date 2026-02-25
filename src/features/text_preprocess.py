import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


def fit_and_apply_label_encoder(
    train_df,
    validation_df,
    label_column,
):
    label_encoder = LabelEncoder()

    train_df = train_df.copy()
    validation_df = validation_df.copy()

    train_df["label"] = label_encoder.fit_transform(train_df[label_column].astype(str))
    validation_df["label"] = label_encoder.transform(validation_df[label_column].astype(str))

    label_names = list(label_encoder.classes_)
    id_to_label = {index: label_names[index] for index in range(len(label_names))}
    label_to_id = {label: index for index, label in id_to_label.items()}

    return train_df, validation_df, label_encoder, id_to_label, label_to_id


def compute_class_weights(
    train_df,
    encoded_label_column="label",
    method="inv_freq",
    epsilon=1e-6,
):
    counts = train_df[encoded_label_column].value_counts().sort_index().values.astype(np.float32)

    if method == "inv_freq":
        weights = counts.sum() / (len(counts) * (counts + epsilon))
    elif method == "sqrt_inv":
        weights = np.sqrt(counts.sum() / (len(counts) * (counts + epsilon)))
    else:
        raise ValueError(f"Unknown class weight method: {method}")

    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float)