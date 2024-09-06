import argparse
from pathlib import Path


def get_args():
    # fmt: off
    description = "Get baseline datasets and their respective experiments.txt file"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output_dir", type=Path, help="Directory to save the JSONL files and the TXT experiments file.")
    parser.add_argument("--input_filepath", type=Path, help="Dataset path to create baselines on.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()


if __name__ == "__main__":
    main()
