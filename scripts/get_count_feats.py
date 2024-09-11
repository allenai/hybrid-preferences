import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import random

from src.feature_extractor import get_all_features, FeatureExtractor

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Sample count features for a specific dataset")
    parser.add_argument("--input_path", type=Path, help="JSONL path to the dataset with fields containing all features.")
    parser.add_argument("--output_dir", type=Path, help="Path to store the sampled outputs.")
    parser.add_argument("--create_experiments_file", type=Path, default=None, help="Store all generated outputs and their hashes in this experiments file.")
    parser.add_argument("--n_instances", type=Path, default=200, help="Number of training instances to create.")
    parser.add_argument("--random_seed", type=int, default=None, help="Set random seed.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    logging.info(f"Setting random seed to {args.random_seed}")
    random.seed(args.random_seed)

    all_features = get_all_features(n_bins=3)
    df = pd.read_json(args.input_path, lines=True)

    # Create a dictionary of features and the `id` of instances that contain it
    feat_instance_map: dict[str, list[str]] = {}
    for feature_str in all_features:
        key, params = FeatureExtractor.parse_feature(feature_str)
        if "min_val" in params or "max_val" in params:
            min_val, max_val = params["min_val"], params["max_val"]
            filtered_df = df[(df[key] >= min_val) & (df[key] <= max_val)]
        elif "analyzer_closed_set" in feature_str:
            feature_name, constraints = params["feature_name"], params["constraints"]
            filtered_df = df[df[feature_name].apply(lambda x: constraints in x)]
        elif "analyzer_scalar" in feature_str:
            feature_name, value = params["feature_name"], params["value"]
            filtered_df = df[df[feature_name] == value]
        elif "analyzer_open_set" in feature_str:
            feature_name = params["feature_name"]
            filtered_df = df[df[feature_name].apply(lambda x: len(x) > 0)]
            breakpoint()

        if len(filtered_df) > 0:
            feat_instance_map[feature_str] = filtered_df["id"].to_list()


if __name__ == "__main__":
    main()
