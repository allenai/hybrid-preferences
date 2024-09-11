import sys
import argparse
import logging
from pathlib import Path

from src.feature_extractor import get_all_features

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Sample count features for a specific dataset")
    parser.add_argument("--input_file", type=Path, help="JSONL path to the dataset with fields containing all features.")
    parser.add_argument("--output_dir", type=Path, help="Path to store the sampled outputs.")
    parser.add_argument("--create_experiments_file", type=Path, default=None, help="Store all generated outputs and their hashes in this experiments file.")
    parser.add_argument("--n_instances", type=Path, default=200, help="Number of training instances to create.")
    parser.add_argument("--random_seed", type=int, default=None, help="Set random seed.")
    # fmt: on
    return parser.parse_args()


def main():
    all_features = get_all_features(n_bins=3)
    breakpoint()


if __name__ == "__main__":
    main()
