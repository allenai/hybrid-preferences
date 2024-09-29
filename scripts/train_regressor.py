import argparse
import logging
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from scripts.get_count_feats import generate_instances
from src.feature_extractor import get_all_features


def get_args():
    # fmt: off
    description = """Train a regressor using the counts-based features

In order to get the training data, you need to run the following command:

```
# Assuming you want helpsteer2's count features
DATASET=helpsteer2 python3 scripts/fetch_evals_rewardbench.py \
    --output_path data/$DATASET-counts-runs.csv \
    --experiment_prefix rm-eval-$DATASET-count \
    --feature_counts_dir data/$DATASET_count_feats/counts/ \
    --dataset_total_size 10160
```

The value passed to `--output_path` is the `--input_path` for this command.
"""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=description)
    parser.add_argument("--input_path", type=Path, required=True, help="Path to the full training dataset (the dev dataset will be extracted from here).")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the features as a JSONL file and the model as a PKL file.")
    parser.add_argument("--model", choices=["lightgbm", "linear", "quadratic"], default="linear", help="Model to use for training the regressor.")
    parser.add_argument("--log_level", default="DEBUG", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    parser.add_argument("--simulator_reference", default=None, help="Path to the 'all-features.jsonl' file to simulate data points.")
    parser.add_argument("--simulator_n_instances", type=int, default=100, help="Number of instances for the simulator.")
    parser.add_argument("--simulator_n_train_samples", type=int, default=7000, help="Number of train samples for each simulated instance.")
    parser.add_argument("--simulator_output_dir", type=Path, default=Path("data/simulator"), help="Directory to save the simulated swaps.")
    parser.add_argument("--simulator_max_budget", default=None, help="If set, will remove instances that exceed the max budget.")
    parser.add_argument("--id_col", type=str, default="id", help="Name of the id column.")
    parser.add_argument("--text_col", type=str, default="text", help="Name of the text column.")
    parser.add_argument("--response_a_col", type=str, default="completion_a", help="Name of the response A column.")
    parser.add_argument("--response_b_col", type=str, default="completion_b", help="Name of the response A column.")
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

    input_df = pd.read_csv(args.input_path).dropna()
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
        "quadratic": train_quadratic_regressor,
    }

    if args.model not in models:
        msg = f"Unknown model: {args.model}"
        logging.error(msg)
        raise ValueError(msg)

    train_fn = models.get(args.model)
    model, results = train_fn(X_train, X_test, y_train, y_test)
    logging.info(f"Regression results for {args.model} ({model}): {results}")

    # Training curve
    logging.info("Computing the train curve...")
    pct_of_train = [0.25, 0.50, 0.75, 1]
    for pct in pct_of_train:
        num_train = int(len(X_train) * pct)
        _, scores = train_fn(X_train[:num_train], X_test, y_train[:num_train], y_test)
        logging.info(f"Performance at {pct:.2%} of train samples: {scores}")

    if args.model == "linear":
        logging.info("*** Feature importance ***")
        feat_impt_df = pd.DataFrame(
            {"feat": model.feature_names_in_, "coef": model.coef_}
        )
        print("Top-5 and bottom-5 features")
        sorted_feat_impt = feat_impt_df.sort_values(by="coef", ascending=False)
        table_kwargs = {"tablefmt": "github", "index": False}
        print(sorted_feat_impt.head(5).to_markdown(**table_kwargs))
        print(sorted_feat_impt.tail(5).to_markdown(**table_kwargs))

    if args.simulator_reference:
        logging.info("*** Simulation proper ***")
        ref_df = pd.read_json(args.simulator_reference, lines=True)

        ref_df = ref_df.rename(
            columns={
                args.id_col: "id",
                args.text_col: "prompt",
                args.response_a_col: "completion_a",
                args.response_b_col: "completion_b",
            }
        )

        sim_df = pd.DataFrame(
            generate_instances(
                df=ref_df,
                n_train_instances=args.simulator_n_instances,
                n_samples=args.simulator_n_train_samples,
                output_dir=args.simulator_output_dir,
            )
        ).transpose()

        sim_df["predicted"] = model.predict(sim_df)
        sim_df["uuid"] = sim_df.index.str.extract(r"ID__(\w+)__")[0].to_list()
        sim_df["budget"] = (
            sim_df.index.str.extract(r"SWAPS_(\d+)")[0].astype(int).to_list()
        )
        sim_df = sim_df.sort_values(by="predicted", ascending=False)

        if args.simulator_max_budget:
            logging.info(f"Removing instances that exceed {args.simulator_max_budget}")
            sim_df = sim_df[sim_df["budget"] <= args.simulator_max_budget]

        sim_results_path = Path(args.simulator_output_dir) / "simulation_results.csv"
        sim_df.to_csv(sim_results_path)
        logging.info(f"Saving files to {sim_results_path}")
    else:
        logging.warning(
            "No value passed in --simulator_reference, will not run simulator."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coeff_output_path = output_dir / "coef.jsonl"
    if args.model == "linear":
        logging.info(f"Saving model coefficients to {coeff_output_path}")
        feat_impt_df.to_json(coeff_output_path, lines=True, orient="records")
    model_output_path = output_dir / "model.pkl"
    logging.info(f"Saving model to {model_output_path}")
    joblib.dump(model, model_output_path)

    if args.model == "quadratic":
        poly_output_path = output_dir / "poly.pkl"
        poly = results.get("poly")
        joblib.dump(poly, poly_output_path)


def train_linear_regressor(X_train, X_test, y_train, y_test, log_linear: bool = False):
    model = LinearRegression()
    if log_linear:
        # Log-linear
        X_train = np.log1p(X_train)
        X_test = np.log1p(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    return model, {"mse": mse, "rmse": rmse}


def train_quadratic_regressor(
    X_train, X_test, y_train, y_test, log_linear: bool = False
):
    poly = PolynomialFeatures(degree=2)

    if log_linear:
        X_train = np.log1p(X_train)
        X_test = np.log1p(X_test)

    # Transform the features
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    return model, {"mse": mse, "rmse": rmse, "poly": poly}


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
