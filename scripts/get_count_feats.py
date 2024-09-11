import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import random
import uuid
import json
from tqdm import tqdm

from src.feature_extractor import get_all_features, FeatureExtractor
from scripts.apply_data_model import convert_to_dpo_format

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

pd.options.mode.chained_assignment = None


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Sample count features for a specific dataset")
    parser.add_argument("--input_path", type=Path, help="JSONL path to the dataset with fields containing all features.")
    parser.add_argument("--output_dir", type=Path, help="Path to store the sampled outputs.")
    parser.add_argument("--create_experiments_file", type=Path, default=None, help="Store all generated outputs and their hashes in this experiments file.")
    parser.add_argument("--n_train_instances", type=int, default=200, help="Number of regression training instances to create.")
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples to create for DPO.")
    parser.add_argument("--random_seed", type=int, default=None, help="Set random seed.")
    parser.add_argument("--id_col", type=str, default="id", help="Name of the id column.")
    parser.add_argument("--text_col", type=str, default="text", help="Name of the text column.")
    parser.add_argument("--response_a_col", type=str, default="completion_a", help="Name of the response A column.")
    parser.add_argument("--response_b_col", type=str, default="completion_b", help="Name of the response A column.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    logging.info(f"Setting random seed to {args.random_seed}")
    random.seed(args.random_seed)

    output_dir = Path(args.output_dir)
    counts_dir = output_dir / "counts"
    counts_dir.mkdir(parents=True, exist_ok=True)
    swaps_dir = output_dir / "swaps"
    swaps_dir.mkdir(parents=True, exist_ok=True)

    all_features = get_all_features(n_bins=3)
    df = pd.read_json(args.input_path, lines=True)

    # Normalize column names
    df = df.rename(
        columns={
            args.id_col: "id",
            args.text_col: "prompt",
            args.response_a_col: "completion_a",
            args.response_b_col: "completion_b",
        }
    )

    if args.n_samples:
        logging.info("Sampling original dataset")
        df = df.sample(n=args.n_samples).reset_index(drop=True)

    logging.info("Creating feature map...")
    # Create a dictionary of features and the `id` of instances that contain it
    feat_instance_map: dict[str, list[str]] = {}
    for feature_str in all_features:
        instances = get_instances(df, feature_str=feature_str)
        feat_instance_map[feature_str] = instances if len(instances) > 0 else []

    budgets = random.sample(range(1, len(df) + 1), args.n_train_instances)

    logging.info("Getting subsets for each budget...")
    tags = []
    uuids = [uuid.uuid4().hex for _ in range(len(budgets))]
    for id, budget in tqdm(zip(uuids, budgets)):
        instances_to_swap = run_knapsack(capacity=budget, items=feat_instance_map)
        to_swap_df = df[df["id"].isin(instances_to_swap)].reset_index(drop=True)

        budget_instance_map = {}
        for feature_str in all_features:
            instances = get_instances(to_swap_df, feature_str)
            budget_instance_map[feature_str] = len(instances)

        tag = f"{id}__BUDGET_{budget}"

        # Save the swaps
        df_swapped = df.copy(deep=True)
        df_swapped["pref"] = df_swapped.apply(
            lambda row: (
                row["pref_human"]
                if row["id"] in instances_to_swap
                else row["pref_gpt4"]
            ),
            axis=1,
        )
        df_swapped["is_swapped"] = df["id"].apply(lambda x: x in instances_to_swap)
        annotations = df_swapped.to_dict(orient="records")
        converted_annotations = []
        for annotation in annotations:
            converted_instance = convert_to_dpo_format(annotation, annotation["pref"])

        # Save the budget
        counts_outfile = counts_dir / f"regressor_feats_{tag}.json"
        with counts_outfile.open("w") as file:
            json.dump(budget_instance_map, file, indent=4)

        # Save the tag file to create the experiments.txt later

    # budgets += [0, len(df)]  # Add lower and upper-bounds
    breakpoint()


def get_instances(df: "pd.DataFrame", feature_str: str) -> list[str]:
    key, params = FeatureExtractor.parse_feature(feature_str)
    if "min_val" in params or "max_val" in params:
        min_val, max_val = params["min_val"], params["max_val"]

        if key in ("prompt_len", "token_len_diff", "len_shorter", "len_longer"):
            df[key] = df[key].rank(pct=True)

        filtered_df = df[(df[key] >= min_val) & (df[key] <= max_val)]
    elif "analyzer_closed_set" in feature_str:
        feature_name, constraints = params["feature_name"], params["constraints"]
        filtered_df = df[df[feature_name].apply(lambda x: constraints in x)]
    elif "analyzer_scalar" in feature_str:
        feature_name, value = params["feature_name"], params["value"]
        filtered_df = df[df[feature_name] == value]
    elif "analyzer_open_set" in feature_str:
        feature_name = params["feature_name"]
        filtered_df = df[df[feature_name].apply(lambda x: x is not None and len(x) > 0)]

    return filtered_df["id"].to_list()


def run_knapsack(capacity: int, items: dict[str, list[str]]) -> list[str]:
    knapsack = []
    total_length = 0

    # Shuffle the keys of the dictionary
    keys = list(items.keys())
    random.shuffle(keys)

    for key in keys:
        item_list = items[key]
        # Shuffle the items within each key
        random.shuffle(item_list)

        for item in item_list:
            if total_length + 1 > capacity:
                return knapsack

            if item not in knapsack:
                knapsack.append(item)
                total_length += 1

    return knapsack


if __name__ == "__main__":
    main()
