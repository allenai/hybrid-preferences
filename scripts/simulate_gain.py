import argparse
from pathlib import Path
from src.feature_extractor import FeatureExtractor


def get_args():
    # fmt: off
    description = "Simulate a dataset when doing partial swaps and compute the predicted performance"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input_path", type=Path, required=True, help="Path to the features.jsonl file for a given dataset.")
    # fmt: on
    pass


def main():
    pass


if __name__ == "__main__":
    main()
