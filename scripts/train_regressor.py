import argparse
import logging
import sys
from io import BytesIO
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

from beaker import Beaker
from scripts.fetch_evals_rewardbench import \
    fetch_evals_rewardbench as fetch_results
from src.simulator import Simulator


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input_mixes_dir", type=Path, help="Directory containing all the JSONL subset files.")
    parser.add_argument("--feats_dataset_id", type=str, default="01J7C8W3ZNQX9HEH4NSSKB8H3B", help="Beaker ID containing the extracted lexical features and metadata features.")
    parser.add_argument("--beaker_workspace", default="ai2/ljm-oe-adapt", help="Beaker workspace to fetch experiments.")
    parser.add_argument("--experiment_prefix", default="rm-eval-", help="Prefix for experiments to fetch.")
    parser.add_argument("--experiments_file", default=None, type=Path, help="Path to a TXT file containing a list that maps an experiment to the features.")
    parser.add_argument("--use_count_feats", action="store_true", default=False, help="If set, will transform features using count-based features.")
    parser.add_argument("--model", choices=["lightgbm", "linear"], default="linear", help="Model to use for training the regressor.")
    parser.add_argument("--log_level", default="DEBUG", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
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

    beaker = Beaker.from_env(default_workspace=args.beaker_workspace)
    try:
        account = beaker.account.whoami()
    except Exception as e:
        logging.error(f"Please authenticate using `beaker account login`: {e}")
        raise
    else:
        logging.info(f"Logged-in as {account.name} ({account.email})")

    results_df = fetch_results(
        beaker=beaker,
        beaker_workspace=args.beaker_workspace,
        experiment_prefix=args.experiment_prefix,
        experiments_file=args.experiments_file,
    )
    logging.debug(f"Found {len(results_df)} results!")
    # fmt: off
    if args.feats_dataset_id:
        feats_df = pd.read_json(BytesIO(b"".join(beaker.dataset.stream_file(args.feats_dataset_id, "features.jsonl"))),lines=True)
        logging.debug(f"Dataset contains {len(feats_df)} instances with {len(feats_df.columns)} features!")
    else:
        feats_df = None
    # fmt: on

    # Get columns that are features (by default, these are binary columns)
    modeling_df = results_df[results_df.columns[results_df.isin([0, 1]).all()]]

    if args.use_count_feats:
        logging.debug("Transforming features to use counts")
        modeling_df = []

    logging.info("*** Modeling proper ***")
    X = modeling_df
    y = results_df["Overall"].astype(float)
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

    logging.info("*** Simulation proper ***")
    simulator = Simulator(
        model=model,
        feat_list=X_train.columns,
        precomputed_features=feats_df,
    )
    simulated_feats = simulator.sample_combinations(n=3_000)
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
