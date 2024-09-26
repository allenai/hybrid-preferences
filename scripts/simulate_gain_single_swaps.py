import sys
import logging
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from scripts.get_count_feats import get_instances, get_all_features

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
    # fmt: off
    description = "Simulate a dataset using a quadratic regressor and get the gain."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input_path", type=Path, required=True, help="Path to the features.jsonl file for a given dataset.")
    parser.add_argument("--feature_name", type=str, required=True, help="Feature name to simulate")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the model PKL file."),
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    input_df = pd.read_json(args.input_path, lines=True)
    model = joblib.load(args.model_path)
    feat_ext = (
        joblib.load(args.model_path.parent / "poly.pkl")
        if "quadratic" in str(args.model_path)
        else None
    )


if __name__ == "__main__":
    main()
