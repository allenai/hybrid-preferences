import argparse
from pathlib import Path

from src.feature_extractor import get_all_features


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Sample count features for a specific dataset")
    parser.add_argument("--input_file", type=Path, help="JSONL path to the dataset with fields containing all features.")
    parser.add_argument("--output_dir", type=Path, help="Path to store the sampled outputs.")
    parser.add_argument("--create_experiments_file", type=Path, default=None, help="Store all generated outputs and their hashes in this experiments file.")
    # fmt: on


def main():
    all_features = get_all_features(n_bins=3)


if __name__ == "__main__":
    main()
