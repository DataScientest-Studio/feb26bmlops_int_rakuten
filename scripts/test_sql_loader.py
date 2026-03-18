"""
Test script to verify loading data from PostgreSQL using load_train_validation_sql.

Usage:
    python scripts/test_sql_loader.py --step 1
    python scripts/test_sql_loader.py --step 2 --db-url postgresql://postgres:postgres@localhost:5432/dst_db
    python scripts/test_sql_loader.py --step 1 --sample 0.1
"""
import argparse
import sys

DEFAULT_DB_URL = "postgresql://postgres:postgres@localhost:5432/dst_db"
TEXT_COLUMN = "text_stripped"
LABEL_COLUMN = "prdtypecode"


def main():
    parser = argparse.ArgumentParser(description="Test SQL data loader")
    parser.add_argument("--step", type=int, required=True, help="Load rows where step <= this value")
    parser.add_argument("--db-url", default=DEFAULT_DB_URL, help="SQLAlchemy database URL")
    parser.add_argument("--sample", type=float, default=None, help="Fraction of data to sample (0, 1]")
    args = parser.parse_args()

    print(f"Connecting to: {args.db_url}")
    print(f"Loading data with step <= {args.step}")
    if args.sample:
        print(f"Sampling {args.sample * 100:.1f}% of rows")

    try:
        from src.data.loaders import load_train_validation_sql
    except ImportError as e:
        print(f"ERROR: Could not import loader: {e}")
        print("Run this script from the project root: python scripts/test_sql_loader.py")
        sys.exit(1)

    try:
        train_df, val_df = load_train_validation_sql(
            db_url=args.db_url,
            step=args.step,
            text_column=TEXT_COLUMN,
            label_column=LABEL_COLUMN,
            sample_number=args.sample,
            seed=42,
        )
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        sys.exit(1)

    print("\n--- Results ---")
    print(f"Train rows:      {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")

    if len(train_df) == 0 or len(val_df) == 0:
        print("\nWARNING: One or both splits are empty. Check that the `step` and `split` columns exist in the table.")
        sys.exit(1)

    print(f"\nTrain label distribution ({LABEL_COLUMN}):")
    print(train_df[LABEL_COLUMN].value_counts().to_string())

    print(f"\nValidation label distribution ({LABEL_COLUMN}):")
    print(val_df[LABEL_COLUMN].value_counts().to_string())

    print(f"\nSample train rows ({TEXT_COLUMN}):")
    for _, row in train_df.head(3).iterrows():
        text_preview = row[TEXT_COLUMN][:80].replace("\n", " ")
        print(f"  [{row[LABEL_COLUMN]}] {text_preview}...")

    print("\nSUCCESS: Data loaded correctly from SQL.")


if __name__ == "__main__":
    main()
