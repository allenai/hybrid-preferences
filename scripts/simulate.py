import argparse
import logging
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.get_count_feats import get_all_features, get_instances
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
    parser.add_argument("--print_latex", action="store_true", default=False, help="Print LaTeX table.")
    parser.add_argument("--n_trials", type=int, default=3, help="Number of trials to run the simulator.")
    parser.add_argument("--flip_size", type=int, default=100, help="Number of swaps for actual simulation.")
    parser.add_argument("--sort", action="store_true", default=False, help="If set, sort by gain.")
    parser.add_argument("--sim_type", choices=["dim_only", "actual"])
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    model, feat_ext = load_model(args.model_path)
    features = get_all_features()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calc_instance_freq(args.input_path, args.output_dir)

    if args.sim_type == "dim_only":
        sim_dim_only(
            input_path=args.input_path,
            swap_pct=[0, 0.01, 0.25, 0.5, 0.75, 1.0],
            output_dir=args.output_dir,
            features=features,
            model=model,
            feat_ext=feat_ext,
            n_trials=args.n_trials,
            sort=args.sort,
            print_latex=args.print_latex,
        )
    elif args.sim_type == "actual":
        sim_actual(
            input_path=args.input_path,
            output_dir=args.output_dir,
            features=features,
            model=model,
            feat_ext=feat_ext,
            n_trials=args.n_trials,
            sort=args.sort,
            flip_size=args.flip_size,
            print_latex=args.print_latex,
        )


def calc_instance_freq(input_path: Path, output_dir: Path):
    df = pd.read_json(input_path, lines=True)
    feat_counts = get_feat_counts(df)
    counts_df = pd.DataFrame([feat_counts]).transpose().reset_index()
    counts_df["index"] = counts_df["index"].apply(lambda x: fmt_prettyname(x))
    counts_df.to_csv(output_dir / "counts.csv", index=False)


def sim_dim_only(
    input_path: Path,
    output_dir: Path,
    swap_pct: list[float],
    features: list[str],
    model,
    feat_ext,
    n_trials: int = 3,
    sort: bool = True,
    print_latex: bool = False,
):
    for random_swaps in swap_pct:
        logging.info(f"Simulating {random_swaps*100}% swaps!")
        n = len(features)
        n_trials = 1 if random_swaps == 0 else n_trials

        df = pd.DataFrame(
            {"feature": [f"BASELINE_{random_swaps}"] + get_all_features()}
        )
        gains = []
        for trial in range(n_trials):
            logging.info(f"Running trial: {trial}")
            baseline_vector = (
                np.zeros(n)
                if random_swaps == 0
                else get_baseline(input_path, random_swaps)
            )
            gdf = pd.DataFrame(
                np.vstack([baseline_vector, baseline_vector + np.eye(n, dtype=int)]),
                columns=features,
            )

            preds = np.array(model.predict(feat_ext.transform(gdf)))
            baseline_score = preds[0]
            gain = preds - baseline_score
            gains.append(gain)

        avg_gain = np.array(gains).mean(axis=0)
        df["gain"] = avg_gain

        df["feature"] = df["feature"].apply(lambda x: fmt_prettyname(x))
        if sort:
            df = df.sort_values(by="gain", ascending=False)
        df = df.reset_index(drop=True)
        if print_latex:
            print(df.to_latex(index=False))
        df.to_csv(output_dir / f"simulated_{random_swaps}.csv", index=False)


def sim_actual(
    input_path: Path,
    output_dir: Path,
    features: list[str],
    model,
    feat_ext,
    flip_size: int = 100,
    n_trials: int = 3,
    sort: bool = False,
    print_latex: bool = False,
):
    input_df = pd.read_json(input_path, lines=True)
    baseline = model.predict(feat_ext.transform([np.zeros(len(features))]))[0]

    all_gains = []
    for trial in range(n_trials):
        logging.info(f"*** Running trial {trial} ***")
        vectors = []
        for feature_str in features:
            # Add +100
            instances_to_swap = get_ids(input_df, feature_str, flip_size=flip_size)
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

            n_actual_swaps = df_swapped["is_swapped"].sum()
            if n_actual_swaps < flip_size:
                logging.info(
                    f"Feature {feature_str} has < {flip_size} ids. Swapped {n_actual_swaps}."
                )
            vector = get_feat_counts(df_swapped[df_swapped["is_swapped"]])
            vectors.append(vector)

        preds = model.predict(feat_ext.transform(pd.DataFrame(vectors)))
        gains = preds - baseline
        all_gains.append(gains)

    gain_df = pd.DataFrame(
        {"feature": features, "gain": np.array(all_gains).mean(axis=0)}
    )

    gain_df["feature"] = gain_df["feature"].apply(lambda x: fmt_prettyname(x))
    if sort:
        gain_df = gain_df.sort_values(by="gain", ascending=False)
    if print_latex:
        print(gain_df.to_latex())

    output_path = output_dir / f"gain_{flip_size}.csv"
    gain_df.to_csv(output_path, index=False)


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


def get_baseline(input_path, pct_swaps):
    df = pd.read_json(input_path, lines=True)
    num_swaps = int(len(df) * pct_swaps)
    instances_to_swap = df["id"].sample(num_swaps).to_list()

    df_swapped = df.copy(deep=True)
    df_swapped["pref"] = df_swapped.apply(
        lambda row: (
            row["pref_human"] if row["id"] in instances_to_swap else row["pref_gpt4"]
        ),
        axis=1,
    )
    df_swapped["is_swapped"] = df["id"].apply(lambda x: x in instances_to_swap)
    feat_counts = get_feat_counts(df_swapped[df_swapped["is_swapped"]])
    return np.array(list(feat_counts.values()))


def get_feat_counts(df: pd.DataFrame) -> dict[str, int]:
    all_features = get_all_features()
    budget_instance_map = {}
    for feature_str in all_features:
        instances = get_instances(df, feature_str)
        budget_instance_map[feature_str] = len(instances)

    return budget_instance_map


def get_ids(df: pd.DataFrame, feature_str: str, flip_size: int) -> dict[int, list[str]]:
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
        ids = df[df[feature_name].apply(lambda x: x is not None and len(x) > 0)]["id"]
    else:
        raise ValueError(f"Unknown feature: {feature_str}")

    ids = ids.to_list()
    sample_ids = random.sample(ids, min(flip_size, len(ids)))
    return sample_ids


if __name__ == "__main__":
    main()
