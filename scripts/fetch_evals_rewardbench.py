import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

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
    parser.add_argument("--output_file", type=Path, help="CSV Filepath to save output features and category scores.")
    parser.add_argument("--beaker_workspace", default="ai2/ljm-oe-adapt", help="Beaker workspace to fetch experiments.")
    parser.add_argument("--experiment_prefix", default="rm-eval-", help="Prefix for experiments to fetch.")
    parser.add_argument("--experiments_file", default=None, type=Path, help="Path to a TXT file containing a list that maps an experiment to the features.")
    parser.add_argument("--gpt4_threshold_score", type=float, default=0.658, help="GPT-4 threshold score to create binary labels")
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

    experiments = [
        experiment
        for experiment in beaker.workspace.experiments(
            args.beaker_workspace,
            match=args.experiment_prefix,
        )
        if is_done(experiment)
    ]
    logging.info(
        f"Found {len(experiments)} experiments that match '{args.experiment_prefix}'"
    )

    # The scores saved on Beaker are subset scores.
    # Let's keep them, but let's also compute the category and overall scores.
    subset_scores: dict[str, dict[str, float]] = {
        experiment.name: beaker.experiment.metrics(experiment)
        for experiment in experiments
    }
    df_subset_scores = pd.DataFrame(subset_scores).transpose().drop(columns=["model"])
    logging.info("Computing category scores...")
    df_category_scores = get_category_scores(df_subset_scores).sort_values(
        by="Overall",
        ascending=False,
    )

    # Turn features into a binary matrix
    df_feats = get_features(
        df_category_scores.reset_index().rename(columns={"index": "experiment"}),
        col_name="experiment",
        experiments_file=args.experiments_file,
    )

    if args.experiments_file:
        logging.info("Will attempt merge via feature hash")

        def extract_hash(string):
            match = re.search(r"FEATS_(.*?)_SWAPS", string)
            return match.group(1) if match else None

        # fmt: off
        df_feats["hash"] = df_feats.index.to_series().apply(extract_hash)
        df_category_scores["hash"] = df_category_scores.index.to_series().apply(extract_hash)
        df_subset_scores["hash"] = df_subset_scores.index.to_series().apply(extract_hash)
        # fmt: on
        overall_df = pd.merge(
            df_feats, df_category_scores, how="inner", on="hash"
        ).merge(df_subset_scores, how="inner", on="hash")

    else:
        overall_df = pd.merge(
            df_feats,
            df_category_scores,
            left_index=True,
            right_index=True,
        ).merge(df_subset_scores, left_index=True, right_index=True)

    # Cleanup dataframe for easier viewing
    meta = ["model_type", "chat_template"]
    cols = meta + [col for col in overall_df.columns if col not in meta]
    overall_df = overall_df[cols]

    # Create labels based on the GPT-4 threshold score
    thresh = args.gpt4_threshold_score
    logging.info(f"Creating labels in column 'label' with GPT-4 threshold '{thresh}'")
    overall_df["label"] = (overall_df["Overall"] > thresh).astype(int)

    overall_df.to_csv(args.output_file)
    logging.info(f"Saved on {args.output_file}")


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
            data = f.read().splitlines()

        for d in data:
            experiment_id, feature_set = d.split("::")
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
