import sys
import logging
import random
import argparse
from pathlib import Path


import pandas as pd

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
    # fmt: off
    description = "Select the best instances to swap to human annotations given a budget."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input_path", type=Path, required=True, help="Path to the features.jsonl file for a given dataset."),
    parser.add_argument("--output_dir", type=Path, required=True, help="Path to save the experiments.txt file and the DPO dataset for training.")
    parser.add_argument("--weights_path", type=Path, required=True, help="Path to the regressor coefficients / weights."),
    parser.add_argument("--budget", nargs="*", type=int, required=True, help="Budgets to create swaps for.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    input_df = pd.read_json(args.input_path, lines=True)
    weights_df = pd.read_json(args.weights_path, lines=True)
    # Convert to binary
    binary_feats = convert_to_binary(input_df, feats=weights_df["feat"].to_list())
    # Compute gains


def convert_to_binary(df: pd.DataFrame, feats: list[str]):
    breakpoint()


if __name__ == "__main__":
    main()
