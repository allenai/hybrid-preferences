import argparse
import json
import logging
import random
import sys
import uuid
from pathlib import Path
import tempfile

import joblib
import pandas as pd
from tqdm import tqdm

from scripts.apply_data_model import convert_to_dpo_format
from scripts.get_count_feats import get_instances, generate_instances
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
    parser.add_argument("--sampling_method", default="topk", choices=["topk", "train_based"], help="Type of sampling technique to use at inference time.")
    parser.add_argument("--budgets", nargs="*", type=float, required=True, help="Budget: percentage of the dataset to be routed to humans.")
    parser.add_argument("--n_samples", type=int, default=7000, help="Number of instances per proxy dataset.")
    parser.add_argument("--id_col", type=str, default="id", help="Name of the id column.")
    parser.add_argument("--text_col", type=str, default="text", help="Name of the text column.")
    parser.add_argument("--response_a_col", type=str, default="completion_a", help="Name of the response A column.")
    parser.add_argument("--response_b_col", type=str, default="completion_b", help="Name of the response A column.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    input_df = pd.read_json(args.input_path, lines=True)
    # Normalize column names
    input_df = input_df.rename(
        columns={
            args.id_col: "id",
            args.text_col: "prompt",
            args.response_a_col: "completion_a",
            args.response_b_col: "completion_b",
        }
    )
    model = joblib.load(args.model_path)

    if args.sampling_method == "topk":
        logging.info("*** Using topk approach ***")
        topk_sampling(
            input_df,
            model,
            budgets=args.budgets,
            n_samples=args.n_samples,
            output_dir=Path(args.output_dir),
        )

    if args.sampling_method == "train_based":
        logging.info("*** Using train_based approach ***")
        train_based_sampling(
            input_df,
            model,
            budgets=args.budgets,
            n_samples=args.n_samples,
            output_dir=Path(args.output_dir),
        )


def train_based_sampling(
    input_df,
    model,
    *,
    budgets: list[float],
    n_samples: int,
    output_dir: Path,
    n_instances_per_budget: int = 100,
    store_topk: int = 10,
):
    counts_dir, swaps_dir = prepare_output_dirs(output_dir)
    tags = []
    for budget in budgets:
        logging.info(f"Simulating instances for budget: {budget}")
        if 0 <= budget <= 1:
            budget = int(len(input_df) * budget)

        def mv(src_path: Path, dest_dir: Path):
            src = Path(src_path)
            dest = Path(dest_dir) / src.name
            src.rename(dest)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            sim_df = pd.DataFrame(
                generate_instances(
                    input_df,
                    n_samples=n_samples,
                    budgets=[budget] * n_instances_per_budget,
                    output_dir=tmpdir,
                )
            ).transpose()

            sim_df["predicted"] = model.predict(sim_df)
            sim_df["uuid"] = sim_df.index.str.extract(r"ID__(\w+)__")[0].to_list()
            sim_df["budget"] = budget
            sim_df = sim_df.sort_values(by="predicted", ascending=False)
            top_sim_df = sim_df.head(store_topk)

            # Move files from top_sim_df to actual output directory
            top_uuids = top_sim_df["uuid"].to_list()
            for uuid in top_uuids:
                count_file = list(tmpdir.rglob(f"counts/*{uuid}*"))[0]
                swaps_file = list(tmpdir.rglob(f"swaps/*{uuid}*"))[0]

                mv(src_path=count_file, dest_dir=counts_dir)
                mv(src_path=swaps_file, dest_dir=swaps_dir)
                tags.append(f"{swaps_file.stem}::{count_file.stem}")

            top_sim_df.to_csv(output_dir / "sim_results_budget_{budget}.csv")

    experiments_file = output_dir / "experiments.txt"
    with experiments_file.open("w") as f:
        f.write("\n".join(tags))


def topk_sampling(
    input_df,
    model,
    *,
    budgets: list[float],
    n_samples: int,
    output_dir: Path,
):
    counts_dir, swaps_dir = prepare_output_dirs(output_dir)
    # Compute gains
    weights_df = pd.DataFrame({"feat": model.feature_names_in_, "coef": model.coef_})
    binary_df = convert_to_binary(input_df, features=weights_df["feat"].to_list())
    results = weights_df.set_index("feat")["coef"] * binary_df
    gain_df = input_df.copy(deep=True)
    gain_df["gain"] = results.sum(axis=1)
    gain_df = gain_df.sort_values(by="gain", ascending=False).reset_index(drop=True)

    # Given a budget, get the top-k and compute the cumulative gain
    uuids = [uuid.uuid4().hex for _ in range(len(budgets))]
    tags = []
    budget_instances: dict[str, dict[str, int]] = {}
    for id, budget in zip(uuids, budgets):
        if 0 <= budget <= 1:
            budget = int(len(input_df) * budget)
        logging.info(f"Creating DPO swaps for budget: {budget}")
        instances_to_swap = gain_df[:budget]["id"].to_list()

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
        annotations = df_swapped.to_dict(orient="records")
        converted_annotations = []
        for annotation in annotations:
            if "model_a" not in annotation:
                annotation["model_a"] = ""
            if "model_b" not in annotation:
                annotation["model_b"] = ""
            if "source" not in annotation:
                annotation["source"] = ""
            if "highest_level_degree" not in annotation:
                annotation["highest_level_degree"] = ""
            converted_instance = convert_to_dpo_format(annotation, annotation["pref"])
            if converted_instance is not None:
                converted_annotations.append(converted_instance)

        if n_samples < len(converted_annotations):
            converted_annotations = random.sample(converted_annotations, n_samples)

        gain = gain_df[:budget]["gain"].sum()
        tag = f"ID__{id}__SWAPS_{budget}"

        swaps_outfile = swaps_dir / f"human_datamodel_counts_{n_samples}_{tag}.jsonl"
        with swaps_outfile.open("w") as f:
            for annotation in converted_annotations:
                f.write(json.dumps(annotation) + "\n")

        # Save the budget
        budget_instance_map = {}
        swapped_ids = [eg["id"] for eg in converted_annotations if eg["is_swapped"]]
        swapped_df = input_df[input_df["id"].isin(swapped_ids)].reset_index(drop=True)
        all_features = weights_df["feat"].to_list()
        for feature_str in all_features:
            instances = get_instances(swapped_df, feature_str)
            budget_instance_map[feature_str] = len(instances)

        # Get predicted score
        pred = model.predict(pd.DataFrame([budget_instance_map]))
        logging.info(f"Predicted performance: {pred}")

        counts_outfile = counts_dir / f"regressor_feats_{tag}.json"
        with counts_outfile.open("w") as file:
            json.dump(budget_instance_map, file, indent=4)

        budget_instances[tag] = budget_instance_map

        # Save the tag file to create the experiments.txt later
        tags.append(f"{swaps_outfile.stem}::{counts_outfile.stem}")

    experiments_file = output_dir / "experiments.txt"
    with experiments_file.open("w") as f:
        f.write("\n".join(tags))


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


def prepare_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    counts_dir = output_dir / "counts"
    counts_dir.mkdir(parents=True, exist_ok=True)
    swaps_dir = output_dir / "swaps"
    swaps_dir.mkdir(parents=True, exist_ok=True)
    return counts_dir, swaps_dir


if __name__ == "__main__":
    main()
