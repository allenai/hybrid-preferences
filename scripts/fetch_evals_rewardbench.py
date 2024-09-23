import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from beaker import Beaker, Experiment

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

EXAMPLE_COUNTS = {
    "alpacaeval-easy": 100,
    "alpacaeval-length": 95,
    "alpacaeval-hard": 95,
    "mt-bench-easy": 28,
    "mt-bench-med": 40,
    "mt-bench-hard": 37,
    "math-prm": 984,  # actual length 447, upweighting to be equal to code
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "llmbar-natural": 100,
    "llmbar-adver-neighbor": 134,
    "llmbar-adver-GPTInst": 92,
    "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46,
    "xstest-should-refuse": 250,
    "xstest-should-respond": 154,
    "donotanswer": 136,
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 164,
    "hep-rust": 164,
}

SUBSET_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}


def get_args():
    # fmt: off
    description = "Get results from Beaker that evaluates on RewardBench"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=Path, help="CSV Filepath to save output features and category scores.")
    parser.add_argument("--beaker_workspace", default="ai2/ljm-oe-adapt", help="Beaker workspace to fetch experiments.")
    parser.add_argument("--experiment_prefix", default="rm-eval-", help="Prefix for experiments to fetch.")
    parser.add_argument("--experiments_file", default=None, type=Path, help="Path to a TXT file containing a list that maps an experiment to the features.")
    parser.add_argument("--feature_counts_dir", default=None, type=Path, help="Path to a directory containing JSON files that contain feature counts.")
    parser.add_argument("--gpt4_threshold_score", type=float, default=None, help="GPT-4 threshold score to create binary labels.")
    parser.add_argument("--dataset_total_size", type=int, default=None, help="Number of instances in the original dataset before downsampling.")
    # fmt:on
    return parser.parse_args()


def main():
    args = get_args()
    beaker = Beaker.from_env(default_workspace=args.beaker_workspace)
    try:
        account = beaker.account.whoami()
    except Exception as e:
        logging.error(f"Please authenticate using `beaker account login`: {e}")
        raise
    else:
        logging.info(f"Logged-in as {account.name} ({account.email})")

    overall_df = fetch_evals_rewardbench(
        beaker=beaker,
        beaker_workspace=args.beaker_workspace,
        experiment_prefix=args.experiment_prefix,
        experiments_file=args.experiments_file,
        feature_counts_dir=args.feature_counts_dir,
        gpt4_threshold_score=args.gpt4_threshold_score,
        dataset_total_size=args.dataset_total_size,
    )

    logging.info(f"Saving {len(overall_df)} results to {args.output_path}")
    overall_df.to_csv(args.output_path)
    logging.info(f"Saved on {args.output_path}")


def fetch_evals_rewardbench(
    beaker: Beaker,
    beaker_workspace: str,
    experiment_prefix: str,
    experiments_file: Optional[Path] = None,
    feature_counts_dir: Optional[Path] = None,
    gpt4_threshold_score: Optional[float] = None,
    dataset_total_size: Optional[int] = None,
) -> pd.DataFrame:
    beaker_experiments = beaker.workspace.experiments(
        beaker_workspace,
        match=experiment_prefix,
    )
    experiments = [exp for exp in beaker_experiments if is_done(exp)]
    logging.info(
        f"Found {len(experiments)} experiments that match '{experiment_prefix}'"
    )

    # The scores saved on Beaker are subset scores.
    # Let's keep them, but let's also compute the category and overall scores.
    subset_scores: dict[str, dict[str, float]] = {
        experiment.name: beaker.experiment.metrics(experiment)
        for experiment in tqdm(experiments)
    }
    df_subset_scores = pd.DataFrame(subset_scores).transpose().drop(columns=["model"])
    logging.info("Computing category scores...")
    df_category_scores = get_category_scores(df_subset_scores).sort_values(
        by="Overall",
        ascending=False,
    )

    if feature_counts_dir:
        logging.info("Will read features from a features directory")
        df_category_scores = df_category_scores[
            df_category_scores.index.str.contains("ID")
        ]
        df_subset_scores = df_subset_scores[df_subset_scores.index.str.contains("ID")]
        df_category_scores["uuid"] = df_category_scores.index.to_series().apply(
            lambda x: re.search(r"ID__([a-f0-9]+)__", x).group(1)
        )
        df_category_scores["budget"] = df_category_scores.index.to_series().apply(
            lambda x: re.search(r"SWAPS_(\d+)", x).group(1)
        )
        if dataset_total_size:
            logging.info("Scaling the budget to current dataset size")
            df_category_scores["budget"] = df_category_scores["budget"].apply(
                lambda x: int(int(x) * 7000 / dataset_total_size)
            )
        df_subset_scores["uuid"] = df_subset_scores.index.to_series().apply(
            lambda x: re.search(r"ID__([a-f0-9]+)__", x).group(1)
        )

        feats = []
        for feat_file in feature_counts_dir.glob("*.json"):
            uuid = re.search(r"ID__([a-f0-9]+)__", feat_file.stem).group(1)
            df_feat = (
                pd.read_json(feat_file, typ="dictionary")
                .rename(index=uuid)
                .reset_index()
                .set_index("index")
                .transpose()
            )
            feats.append(df_feat)
        df_feats = pd.concat(feats).reset_index().rename(columns={"index": "uuid"})
        df_scores = df_category_scores.merge(df_subset_scores, on="uuid", how="left")
        overall_df = df_scores.merge(df_feats, on="uuid", how="left").dropna()

    elif experiments_file:
        logging.info("Will attempt merge via feature hash")

        # Turn features into a binary matrix
        df_feats = get_features(
            df_category_scores.reset_index().rename(columns={"index": "experiment"}),
            col_name="experiment",
            experiments_file=experiments_file,
        )

        def extract_hash(string):
            match = re.search(r"FEATS_(.*?)_SWAPS", string)
            return match.group(1) if match else None

        def extract_swaps(string):
            return int(string.split("SWAPS")[1].removeprefix("_"))

        # fmt: off
        df_feats["hash"] = df_feats.index.to_series().apply(extract_hash)
        df_feats["num_swaps"] = df_feats.index.to_series().apply(extract_swaps)
        df_category_scores["hash"] = df_category_scores.index.to_series().apply(extract_hash)
        df_subset_scores["hash"] = df_subset_scores.index.to_series().apply(extract_hash)
        # fmt: on
        overall_df = (
            pd.merge(
                df_feats,
                df_category_scores,
                how="inner",
                on="hash",
            )
            .reset_index()
            .merge(df_subset_scores, how="inner", on="hash")
            .set_index("hash")
        )

    else:
        overall_df = df_category_scores.merge(
            df_subset_scores, left_index=True, right_index=True
        )

    # Cleanup dataframe for easier viewing
    meta = ["model_type", "chat_template"]
    cols = meta + [col for col in overall_df.columns if col not in meta]
    overall_df = overall_df[cols]

    # Create labels based on the GPT-4 threshold score
    thresh = gpt4_threshold_score
    if thresh:
        logging.info(f"Creating labels in 'label' with GPT-4 threshold '{thresh}'")
        overall_df["label"] = (overall_df["Overall"] > thresh).astype(int)

    overall_df = overall_df.sort_values(
        by=["Overall"],
        ascending=False,
    )  # .drop(columns=["index"])
    if "index" in overall_df.columns:
        overall_df = overall_df.drop(columns=["index"])
    overall_df = overall_df[~overall_df.index.duplicated(keep="first")]
    return overall_df


def is_done(experiment: "Experiment") -> bool:
    return True if experiment.jobs[0].status.finalized else False


def get_category_scores(df_subset: "pd.DataFrame") -> "pd.DataFrame":
    category_scores = {}
    for category, subsets in SUBSET_MAPPING.items():
        weights = {k: v for k, v in EXAMPLE_COUNTS.items() if k in subsets}
        category_scores[category] = (df_subset[subsets] * pd.Series(weights)).sum(
            axis=1
        ) / sum(weights.values())
    df_category = pd.DataFrame(category_scores)
    df_category["Overall"] = df_category.mean(axis=1)
    return df_category


def get_features(
    df: "pd.DataFrame",
    col_name: str,
    experiments_file: Optional[Path] = None,
) -> "pd.DataFrame":
    experiment_to_feats: dict[str, list[str]] = {}
    experiments = df[col_name].to_list()

    if not experiments_file:
        logging.info("Deriving features from the experiment names")
        for experiment in experiments:
            features = experiment.split("FEATS_")[-1].split("___")
            experiment_to_feats[experiment] = features

    else:
        logging.info(f"Deriving features from the experiments file: {experiments_file}")
        with open(experiments_file, "r") as f:
            lines = f.read().splitlines()

        for line in lines:
            experiment_id, feature_set = line.split("::")
            experiment_to_feats[experiment_id] = [
                feature.replace("-", "=") for feature in feature_set.split("___")
            ]

    unique_features = set(f for feats in experiment_to_feats.values() for f in feats)
    df_feats = pd.DataFrame(
        [
            {feat: int(feat in feats) for feat in unique_features}
            for feats in experiment_to_feats.values()
        ],
        index=experiment_to_feats.keys(),
    )
    # Sort columns alphabetically
    df_feats = df_feats.reindex(sorted(df_feats.columns), axis=1)
    return df_feats


if __name__ == "__main__":
    main()
