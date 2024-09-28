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
    description = "Simulate a dataset using a quadratic regressor and get the gain."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output_path", type=Path, required=True, help="Path to save the output in a CSV file."),
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the model PKL file."),
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    model = joblib.load(args.model_path)
    feat_ext = (
        joblib.load(args.model_path.parent / "poly.pkl")
        if "quadratic" in str(args.model_path)
        else None
    )
    features = get_all_features()
    n = len(features)
    df = pd.DataFrame(np.vstack([np.zeros(n), np.eye(n, dtype=int)]), columns=features)
    df["pred"] = model.predict(feat_ext.transform(df))
    df["gain"] = df["pred"] - df.loc[0, "pred"]
    df["feature"] = df.iloc[:, :-2].idxmax(axis=1)
    df["feature"].iloc[0] = "Pure synthetic annotations"

    gain_df = df[["feature", "pred", "gain"]].sort_values(by="gain", ascending=False)
    gain_df["gain_log"] = np.log1p(gain_df["gain"] * 1e5)
    gain_df = gain_df[["feature", "gain_log"]].rename(columns={"gain_log": "gain"})

    gain_df["feature"] = gain_df["feature"].apply(lambda x: fmt_prettyname(x))
    gain_df = gain_df.reset_index(drop=True)
    print(gain_df.to_latex(index=False))
    gain_df.to_csv(args.output_path, index=False)


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


if __name__ == "__main__":
    main()
