import sys
import logging
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from src.feature_extractor import FeatureExtractor
from scripts.get_count_feats import get_instances, get_all_features

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
    # fmt: off
    description = "Simulate a dataset when doing partial swaps and compute the predicted performance"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input_path", type=Path, required=True, help="Path to the features.jsonl file for a given dataset.")
    parser.add_argument("--feature_name", type=str, required=True, help="Feature name to simulate")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the model PKL file."),
    parser.add_argument("--n_trials", type=int, default=3, help="Number of trials to run the simulator.")
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

    all_perf_scores = []
    all_perf_gains = []
    all_perf_pct_gains = []
    for trial in range(args.n_trials):
        logging.info(f"*** Running trial {trial+1} ***")
        swap_ids = get_ids(input_df, feature_str=args.feature_name)
        swap_dfs: dict[int, pd.DataFrame] = {}
        for pct, instances_to_swap in swap_ids.items():
            df_swapped = input_df.copy(deep=True)
            df_swapped["pref"] = df_swapped.apply(
                lambda row: (
                    row["pref_human"]
                    if row["id"] in instances_to_swap
                    else row["pref_gpt4"]
                ),
                axis=1,
            )
            df_swapped["is_swapped"] = input_df["id"].apply(
                lambda x: x in instances_to_swap
            )
            swap_dfs[pct] = get_feat_counts(df_swapped[df_swapped["is_swapped"]])

        sim_df = pd.DataFrame(swap_dfs).transpose()
        # Add zeroth row
        cols = sim_df.columns
        zero_row = pd.DataFrame([[0] * len(cols)], columns=cols)
        baseline_perf = model.predict(zero_row)[0]

        # Get scores for this trials
        all_perf_scores.append(model.predict(sim_df))
        all_perf_gains.append(model.predict(sim_df) - baseline_perf)
        all_perf_pct_gains.append(
            [pct_gain(baseline_perf, pred) for pred in model.predict(sim_df)]
        )

    score_report_df = pd.DataFrame(
        {
            "pct": [25, 50, 75, 100],
            "score": np.vstack(all_perf_scores).mean(axis=0),
            "gain": np.vstack(all_perf_gains).mean(axis=0),
            "pct_gain": np.vstack(all_perf_pct_gains).mean(axis=0),
        }
    )
    print(f"\nGain for {args.feature_name} (baseline={baseline_perf})")
    print(score_report_df.to_markdown(tablefmt="github"))


def pct_gain(old, new):
    return ((new - old) / old) * 100


def get_ids(df: pd.DataFrame, feature_str: str) -> dict[int, list[str]]:
    key, params = FeatureExtractor.parse_feature(feature_str)
    if "min_val" in params or "max_val" in params:
        min_val, max_val = params["min_val"], params["max_val"]
        if key in ("prompt_len", "token_len_diff", "len_shorter", "len_longer"):
            df[key] = df[key].rank(pct=True)
        ids = df[(df[key] > min_val) & (df[key] < max_val)]["id"]
    elif "analyzer_closed_set" in feature_str:
        feature_name, constraints = params["feature_name"], params["constraints"]
        ids = df[df[feature_name].apply(lambda x: constraints in x)]["id"]
    elif "analyzer_scalar" in feature_str:
        feature_name, value = params["feature_name"], params["value"]
        ids = df[df[feature_name] == value]["id"]
    elif "analyzer_open_set" in feature_str:
        feature_name = params["feature_name"]
        ids = df[feature_name].apply(lambda x: x is not None and len(x) > 0)["id"]
    else:
        raise ValueError(f"Unknown feature: {feature_str}")

    ids = ids.to_list()
    logging.info(f"Found {len(ids)} instances for '{feature_str}'")
    logging.info("Generating simulated swaps")
    pcts = [0.25, 0.50, 0.75, 1.00]
    swap_ids = {}
    for pct in pcts:
        n_samples = int(len(ids) * pct)
        sample_ids = random.sample(ids, n_samples)
        swap_ids[int(pct * 100)] = sample_ids
        logging.info(f"{pct * 100}%: {len(sample_ids)}")

    return swap_ids


def get_feat_counts(df: pd.DataFrame) -> dict[str, int]:
    all_features = get_all_features()
    budget_instance_map = {}
    for feature_str in all_features:
        instances = get_instances(df, feature_str)
        budget_instance_map[feature_str] = len(instances)

    return budget_instance_map


if __name__ == "__main__":
    main()
