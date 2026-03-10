import argparse
import json

from src.models.predict_model import predict_text


def parse_args():
    parser = argparse.ArgumentParser(description="Run one text prediction from a trained run_id")
    parser.add_argument("--run-id", required=True, help="Training run_id under models/text/<run_id>")
    parser.add_argument("--text", required=True, help="Input text to classify")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max length")
    parser.add_argument(
        "--base-dir",
        default="models/text",
        help="Base directory containing text runs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = predict_text(
        {
            "run_id": args.run_id,
            "text": args.text,
            "max_length": args.max_length,
            "base_dir": args.base_dir,
        }
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
