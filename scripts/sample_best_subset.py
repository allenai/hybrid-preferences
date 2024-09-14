import sys
import logging
import random
import argparse
from pathlib import Path


import pandas as pd
import joblib
from tqdm import tqdm
from src.feature_extractor import FeatureExtractor

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
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the model PKL file."),
    parser.add_argument("--budgets", nargs="*", type=float, required=True, help="Budget: percentage of the dataset to be routed to humans.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    input_df = pd.read_json(args.input_path, lines=True)
    model = joblib.load(args.model_path)

    # Compute gains
    weights_df = pd.DataFrame({"feat": model.feature_names_in_, "coef": model.coef_})
    binary_df = convert_to_binary(input_df, features=weights_df["feat"].to_list())
    results = weights_df.set_index("feat")["coef"] * binary_df
    gain_df = input_df.copy(deep=True)
    gain_df["gain"] = results.sum(axis=1)
    gain_df = gain_df.sort_values(by="gain", ascending=False).reset_index(drop=True)

    # Given a budget, get the top-k and compute the cumulative gain
    for budget in args.budgets:
        breakpoint()
        pass

    # Might also need to get the predicted score from the model


def convert_to_binary(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    binary_cols: dict[str, list[int]] = {}
    logging.info("Getting binary features")
    for feature_str in tqdm(features):
        key, params = FeatureExtractor.parse_feature(feature_str)
        if "min_val" in params or "max_val" in params:
            min_val, max_val = params["min_val"], params["max_val"]
            if key in ("prompt_len", "token_len_diff", "len_shorter", "len_longer"):
                df[key] = df[key].rank(pct=True)
            binary_col = (df[key] > min_val) & (df[key] < max_val)
        elif "analyzer_closed_set" in feature_str:
            feature_name, constraints = params["feature_name"], params["constraints"]
            binary_col = df[feature_name].apply(lambda x: constraints in x)
        elif "analyzer_scalar" in feature_str:
            feature_name, value = params["feature_name"], params["value"]
            binary_col = df[feature_name] == value
        elif "analyzer_open_set" in feature_str:
            feature_name = params["feature_name"]
            binary_col = df[feature_name].apply(lambda x: x is not None and len(x) > 0)
        else:
            raise ValueError(f"Unknown feature: {feature_str}")

        binary_cols[feature_str] = binary_col.astype(int).to_list()

    return pd.DataFrame(binary_cols)


if __name__ == "__main__":
    main()
