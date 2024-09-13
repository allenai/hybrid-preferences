import argparse
import logging
import sys
from pathlib import Path
import random

import lightgbm as lgb
import pandas as pd
import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

from src.simulator import Simulator
from src.feature_extractor import get_all_features
from scripts.get_count_feats import generate_instances
from scripts.apply_data_model import convert_to_dpo_format


def get_args():
    # fmt: off
    description = """Train a regressor using the counts-based features

In order to get the training data, you need to run the following command:

```
# Assuming you want helpsteer2's count features
DATASET=helpsteer2 python3 scripts/fetch_evals_rewardbench.py \
    --output_file data/$DATASET-counts-runs.csv \
    --experiment_prefix rm-eval-$DATASET-count \
    --feature_counts_dir data/$DATASET_count_feats/counts/ \
    --dataset_total_size 10160
```

The value passed to `--output_file` is the `--input_file` for this command.
"""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=description)
    parser.add_argument("--input_file", type=Path, help="Path to the full training dataset (the dev dataset will be extracted from here).")
    parser.add_argument("--model", choices=["lightgbm", "linear"], default="linear", help="Model to use for training the regressor.")
    parser.add_argument("--log_level", default="DEBUG", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    parser.add_argument("--simulator_reference", default=None, help="Path to the 'all-features.jsonl' file to simulate data points.")
    parser.add_argument("--simulator_n_instances", type=int, default=100, help="Number of instances for the simulator.")
    parser.add_argument("--simulator_n_train_samples", type=int, default=7000, help="Number of train samples for each simulated instance.")
    parser.add_argument("--simulator_output_dir", type=Path, default=Path("data/simulator"), help="Directory to save the simulated swaps.")
    parser.add_argument("--random_seed", type=int, default=42, help="Set the random seed.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=getattr(logging, args.log_level),
    )

    input_df = pd.read_csv(args.input_file)
    all_feats = get_all_features()
    modeling_df = input_df[[col for col in input_df.columns if col in all_feats]]

    logging.info("*** Modeling proper ***")
    X = modeling_df
    y = input_df["Overall"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_seed
    )
    logging.info(f"Training size: {len(X_train)}, test_size: {len(X_test)}")

    models: dict[str, callable] = {
        "lightgbm": train_lightgbm_regressor,
        "linear": train_linear_regressor,
    }
    train_fn = models.get(args.model)
    model, results = train_fn(X_train, X_test, y_train, y_test)
    logging.info(f"Regression results for {args.model} ({model}): {results}")

    # Training curve
    logging.info("Computing the train curve...")
    pct_of_train = [0.25, 0.50, 0.75, 1]
    for pct in pct_of_train:
        num_train = int(len(X_train) * pct)
        _, scores = train_fn(X_train[:num_train], X_test, y_train[:num_train], y_test)
        logging.debug(f"Performance at {pct:.2%} of train samples: {scores}")

    logging.info("*** Feature importance ***")
    feat_impt_df = pd.DataFrame(
        {"feat": model.feature_names_in_, "coef": model.coef_}
    ).sort_values(by="coef", ascending=False)
    print("Top-5 and bottom-5 features")
    print(feat_impt_df.head(5).to_markdown(tablefmt="github"))
    print(feat_impt_df.tail(5).to_markdown(tablefmt="github"))

    if args.simulator_reference:
        logging.info("*** Simulation proper ***")
        sim_df = pd.read_json(args.simulator_reference, lines=True)

        budget_instances = generate_instances(
            df=sim_df,
            n_train_instances=args.simulator_n_instances,
            n_samples=args.simulator_n_train_samples,
            output_dir=args.simulator_output_dir,
        )
        breakpoint()
    else:
        logging.info(
            "No value passed in --simulator_reference, will not run simulator."
        )

    # simulator = Simulator(
    #     model=model,
    #     feat_list=X_train.columns,
    #     # precomputed_features=feats_df,
    # )
    # simulated_feats = simulator.sample_combinations(n=3_000)
    breakpoint()


def train_linear_regressor(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    return model, {"mse": mse, "rmse": rmse}


def train_lightgbm_regressor(X_train, X_test, y_train, y_test):
    train_data = lgb.Dataset(X_train, label=y_train, params={"verbose": -1})
    test_data = lgb.Dataset(
        X_test, label=y_test, reference=train_data, params={"verbose": -1}
    )
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting": "gbdt",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "scale_pos_weight": 0.4,
    }
    # Train the model
    model = lgb.train(params, train_data, valid_sets=[test_data])
    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    return model, {"mse": mse, "rmse": rmse}


if __name__ == "__main__":
    main()
