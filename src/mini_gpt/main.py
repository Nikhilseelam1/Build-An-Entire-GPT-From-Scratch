import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Mini-GPT entry point"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "pretrained"],
        required=True,
        help="Run mode: train or pretrained",
    )

    args = parser.parse_args()

    if args.mode == "train":
        subprocess.run(
            [sys.executable, "scripts/train.py"],
            check=True
        )

    elif args.mode == "pretrained":
        subprocess.run(
            [sys.executable, "scripts/load_pretrained.py"],
            check=True
        )


if __name__ == "__main__":
    main()
