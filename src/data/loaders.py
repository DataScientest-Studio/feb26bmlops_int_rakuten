import pandas as pd


def load_train_validation_csv(
    train_csv_path,
    validation_csv_path,
    text_column,
    label_column,
    sample_number=None,
    seed=42,
):
    train_df = pd.read_csv(train_csv_path)
    validation_df = pd.read_csv(validation_csv_path)

    required_columns = [text_column, label_column]
    train_df = train_df[required_columns]
    validation_df = validation_df[required_columns]

    train_df[text_column] = train_df[text_column].astype(str)
    train_df[label_column] = train_df[label_column].astype(str)
    validation_df[text_column] = validation_df[text_column].astype(str)
    validation_df[label_column] = validation_df[label_column].astype(str)

    if sample_number is not None:
        if not isinstance(sample_number, float):
            raise ValueError("sample_number must be a float fraction in (0, 1]")
        if sample_number <= 0 or sample_number > 1:
            raise ValueError("sample_number must be a float fraction in (0, 1]")

        train_n = max(1, int(len(train_df) * sample_number))
        val_n = max(1, int(len(validation_df) * sample_number))

        train_df = train_df.sample(n=train_n, random_state=seed).reset_index(drop=True)
        validation_df = validation_df.sample(n=val_n, random_state=seed).reset_index(drop=True)

    return train_df, validation_df