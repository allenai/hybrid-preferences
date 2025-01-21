import argparse
import json
import logging
import random
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from tqdm import tqdm

from scripts.apply_data_model import convert_to_dpo_format
from scripts.get_count_feats import generate_instances, get_all_features
from scripts.get_count_feats import get_instances
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
    parser.add_argument("--sampling_method", default="topk", choices=["topk", "simulated", "optimal_simulated", "optimal_pos", "optimal_grad"], help="Type of sampling technique to use at inference time.")
    parser.add_argument("--budgets", nargs="*", type=float, required=False, help="Budget: percentage of the dataset to be routed to humans.")
    parser.add_argument("--n_samples", type=int, default=7000, help="Number of instances per proxy dataset.")
    parser.add_argument("--n_simulations", type=int, default=500, help="Number of simulations for a given budget.")
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
    feat_ext = (
        joblib.load(args.model_path.parent / "poly.pkl")
        if "quadratic" in str(args.model_path)
        else None
    )

    if args.sampling_method == "topk":
        logging.info("*** Using 'topk' approach ***")
        assert args.budgets, "Must supply a budget"
        topk_sampling(
            input_df,
            model,
            feat_ext=feat_ext,
            budgets=args.budgets,
            n_samples=args.n_samples,
            optimal=None,
            output_dir=Path(args.output_dir),
        )

    if args.sampling_method == "simulated":
        logging.info("*** Using 'simulated' approach ***")
        assert args.budgets, "Must supply a budget"
        simulated_sampling(
            input_df,
            model,
            feat_ext=feat_ext,
            budgets=args.budgets,
            n_samples=args.n_samples,
            output_dir=Path(args.output_dir),
        )

    if args.sampling_method == "optimal_simulated":
        logging.info("*** Using 'optimal_simulated' approach")
        if not args.budgets:
            logging.info("No budgets provided, using default budgets.")
            budgets = [random.randint(1, len(input_df)) for _ in range(500)]
        else:
            budgets = args.budgets
        simulated_sampling(
            input_df,
            model,
            feat_ext=feat_ext,
            budgets=budgets,
            n_samples=args.n_samples,
            n_instances_per_budget=args.n_simulations,
            output_dir=Path(args.output_dir),
        )

    if args.sampling_method == "optimal_pos":
        logging.info("*** Using 'optimal_pos' approach ***")
        topk_sampling(
            input_df,
            model,
            feat_ext=feat_ext,
            budgets=args.budgets,
            n_samples=args.n_samples,
            optimal="optimal_pos",
            output_dir=Path(args.output_dir),
        )

    if args.sampling_method == "optimal_grad":
        logging.info("*** Using 'optimal_grad' approach ***")
        topk_sampling(
            input_df,
            model,
            feat_ext=feat_ext,
            budgets=args.budgets,
            n_samples=args.n_samples,
            optimal="optimal_grad",
            output_dir=Path(args.output_dir),
        )


def simulated_sampling(
    input_df,
    model,
    *,
    n_samples: int,
    output_dir: Path,
    budgets: Optional[list[float]],
    n_instances_per_budget: int = 500,
    store_topk: int = 10,
    feat_ext=None,
):
    counts_dir, swaps_dir = prepare_output_dirs(output_dir)
    tags = []
    for budget in budgets:
        if 0 <= budget <= 1:
            budget = int(len(input_df) * budget)

        logging.info(f"Simulating instances for budget: {budget}")

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

            input_feats = feat_ext.transform(sim_df) if feat_ext else sim_df
            preds = model.predict(input_feats)
            sim_df["predicted"] = preds
            sim_df["uuid"] = sim_df.index.str.extract(r"ID__(\w+)__")[0].to_list()
            sim_df["budget"] = budget
            sim_df = sim_df.sort_values(by="predicted", ascending=False)
            top_sim_df = sim_df.head(store_topk)
            print(top_sim_df[["uuid", "predicted"]].to_markdown(tablefmt="github"))

            # Move files from top_sim_df to actual output directory
            top_uuids = top_sim_df["uuid"].to_list()
            for uuid in top_uuids:
                count_file = list(tmpdir.rglob(f"counts/*{uuid}*"))[0]
                swaps_file = list(tmpdir.rglob(f"swaps/*{uuid}*"))[0]

                mv(src_path=count_file, dest_dir=counts_dir)
                mv(src_path=swaps_file, dest_dir=swaps_dir)
                tags.append(f"{swaps_file.stem}::{count_file.stem}")

            top_sim_df.to_csv(output_dir / f"sim_results_budget_{budget}.csv")

    experiments_file = output_dir / "experiments.txt"
    with experiments_file.open("w") as f:
        f.write("\n".join(tags))


def topk_sampling(
    input_df,
    model,
    *,
    n_samples: int,
    output_dir: Path,
    budgets: Optional[list[float]] = None,
    optimal: Optional[str] = None,
    feat_ext=None,
):
    counts_dir, swaps_dir = prepare_output_dirs(output_dir)
    # Compute gains
    if feat_ext:
        gain_df = compute_gain_quadratic(input_df, model, feat_ext)
        is_quadratic = True
    else:
        gain_df = compute_gain_linear(input_df, model)
        is_quadratic = False

    # Given a budget, get the top-k and compute the cumulative gain
    uuids = [uuid.uuid4().hex for _ in range(len(budgets))]
    tags = []
    budget_instances: dict[str, dict[str, int]] = {}
    for id, budget in zip(uuids, budgets):
        if 0 <= budget <= 1:
            budget = int(len(input_df) * budget)

        if not optimal:
            logging.info(f"Creating DPO swaps for budget: {budget}")

        instances_to_swap = (
            gain_df[:budget]["id"].to_list()
            if not optimal
            else get_optimal_subset(
                gain_df,
                method=optimal,
                optimal_grad_epsilon=1e-4 if not is_quadratic else 1e-5,
            )
        )

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
        all_features = get_all_features()
        for feature_str in all_features:
            instances = get_instances(swapped_df, feature_str)
            budget_instance_map[feature_str] = len(instances)

        # Get predicted score
        _swap_feats = pd.DataFrame([budget_instance_map])
        feats = _swap_feats if not feat_ext else feat_ext.transform(_swap_feats)
        pred = model.predict(feats)
        logging.info(f"Predicted performance: {pred}")

        counts_outfile = counts_dir / f"regressor_feats_{tag}.json"
        with counts_outfile.open("w") as file:
            json.dump(budget_instance_map, file, indent=4)

        budget_instances[tag] = budget_instance_map

        # Save the tag file to create the experiments.txt later
        tags.append(f"{swaps_outfile.stem}::{counts_outfile.stem}")

        if optimal:
            logging.info("Optimal value passed, will only return the best subset.")
            break

    experiments_file = output_dir / "experiments.txt"
    with experiments_file.open("w") as f:
        f.write("\n".join(tags))


def compute_gain_linear(input_df: pd.DataFrame, model) -> pd.DataFrame:
    weights_df = pd.DataFrame({"feat": model.feature_names_in_, "coef": model.coef_})
    binary_df = convert_to_binary(input_df, features=weights_df["feat"].to_list())
    results = weights_df.set_index("feat")["coef"] * binary_df
    gain_df = input_df.copy(deep=True)
    gain_df["gain"] = results.sum(axis=1)
    gain_df = gain_df.sort_values(by="gain", ascending=False).reset_index(drop=True)
    return gain_df


def compute_gain_quadratic(
    input_df: pd.DataFrame,
    model,
    feat_ext,
    batch_size: int = 1,
) -> pd.DataFrame:
    all_features = get_all_features(n_bins=3)

    def _count_feats(df: pd.DataFrame) -> dict[str, list[str]]:
        feat_instance_map: dict[str, list[str]] = {}
        for feature_str in all_features:
            instances = get_instances(df, feature_str=feature_str)
            feat_instance_map[feature_str] = instances if len(instances) > 0 else []
        return feat_instance_map

    def _batches(lst: list[str], size: int) -> list[list[str]]:
        num_batches = (len(lst) + size - 1) // batch_size
        batches = [lst[i * size : (i + 1) * size] for i in range(num_batches)]
        return batches

    init_df = pd.DataFrame(0, index=range(1), columns=all_features)
    binary_df = convert_to_binary(input_df, features=get_all_features())
    gains = model.predict(feat_ext.transform(binary_df)) - model.predict(
        feat_ext.transform(init_df)
    )
    gain_df = input_df.copy(deep=True)
    gain_df["gain"] = gains
    gain_df = gain_df.sort_values(by="gain", ascending=False)
    return gain_df


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


def get_optimal_subset(
    df: pd.DataFrame,
    method: str,
    gain_col: str = "gain",
    optimal_grad_epsilon=1e-4,
) -> list[str]:
    if method == "optimal_pos":
        # Get instancs with positive gain only
        ids = df[df[gain_col] > 0]["id"].to_list()
    elif method == "optimal_grad":
        # Get instances that contribute a lot to the cumulative gain
        cumulative_gain = 0
        ids = []
        for _, row in df.iterrows():
            if row["gain"] + cumulative_gain > cumulative_gain + optimal_grad_epsilon:
                ids.append(row["id"])
                cumulative_gain += row["gain"]
    else:
        raise ValueError(f"Unknown optimal computation method: {method}")

    logging.info(f"Optimal subset contains {len(ids)} instances")
    return ids


def prepare_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    counts_dir = output_dir / "counts"
    counts_dir.mkdir(parents=True, exist_ok=True)
    swaps_dir = output_dir / "swaps"
    swaps_dir.mkdir(parents=True, exist_ok=True)
    return counts_dir, swaps_dir


if __name__ == "__main__":
    main()
