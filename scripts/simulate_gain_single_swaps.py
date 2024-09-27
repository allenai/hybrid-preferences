import sys
import logging
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from scripts.get_count_feats import get_instances, get_all_features
from sklearn.preprocessing import QuantileTransformer


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

    breakpoint()


if __name__ == "__main__":
    main()
