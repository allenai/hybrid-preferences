import sys
import logging
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from scripts.get_count_feats import get_instances, get_all_features
from src.feature_extractor import FeatureExtractor


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
    # fmt: off
    description = "Simulate a dataset using a trained regressor and get the gain."
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_path", type=Path, required=True, help="Path to the features.jsonl file for a given dataset.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the output in a CSV file."),
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the model PKL file."),
    parser.add_argument("--n_trials", type=int, default=3, help="Number of trials to run the simulator.")
    parser.add_argument("--print_latex", action="store_true", default=False, help="Print LaTeX table.")
    parser.add_argument("--sim_type", choices=["dim_only", "actual"])
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    model, feat_ext = load_model(args.model_path)
    features = get_all_features()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for random_swaps in [0, 0.25, 0.5, 0.75, 1.0]:
        n = len(features)

        baseline_vector = (
            np.zeros(n)
            if random_swaps == 0
            else get_baseline(args.input_path, random_swaps)
        )

        df = pd.DataFrame(
            np.vstack([baseline_vector, np.eye(n, dtype=int)]),
            columns=features,
        )
        df["pred"] = model.predict(feat_ext.transform(df))
        df["gain"] = df["pred"] - df.loc[0, "pred"]
        df["feature"] = df.iloc[:, :-2].idxmax(axis=1)
        df["feature"].iloc[0] = f"BASELINE_{random_swaps}"

        gain_df = df[["feature", "pred", "gain"]].sort_values(
            by="gain", ascending=False
        )
        gain_df["gain_log"] = np.log1p(gain_df["gain"] * 1e5)
        gain_df = gain_df[["feature", "gain_log"]].rename(columns={"gain_log": "gain"})

        gain_df["feature"] = gain_df["feature"].apply(lambda x: fmt_prettyname(x))
        gain_df = gain_df.reset_index(drop=True)
        if args.print_latex:
            print(gain_df.to_latex(index=False))
        gain_df.to_csv(output_dir / f"simulated_{random_swaps}.csv", index=False)


def fmt_prettyname(feature_str: str) -> str:
    key, params = FeatureExtractor.parse_feature(feature_str)
    if "min_val" in params or "max_val" in params:
        min_val, max_val = params["min_val"], params["max_val"]
        key = key.replace("_", " ").title()
        pretty_name = f"{key} \in \{{{min_val}, {max_val}\}}"
    elif "analyzer_closed_set" in feature_str:
        feature_name, constraints = params["feature_name"], params["constraints"]
        pretty_name = f"{feature_name}: {constraints}"
        pretty_name = pretty_name.replace("_", " ").title()
    elif "analyzer_scalar" in feature_str:
        feature_name, value = params["feature_name"], params["value"]
        pretty_name = f"{feature_name}: {value}"
        pretty_name = pretty_name.replace("_", " ").title()
    elif "analyzer_open_set" in feature_str:
        feature_name = params["feature_name"]
        pretty_name = f"{feature_name}"
        pretty_name = pretty_name.replace("_", " ").title()
    else:
        pretty_name = feature_str

    return pretty_name


def load_model(model_path: Path):
    model = joblib.load(model_path)
    feat_ext = (
        joblib.load(model_path.parent / "poly.pkl")
        if "quadratic" in str(model_path)
        else None
    )
    return model, feat_ext


if __name__ == "__main__":
    main()
