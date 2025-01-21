import argparse
import json
import logging
import random
import sys
from inspect import signature
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import root_mean_squared_error

from scripts.sample_best_subset import compute_gain_linear
from scripts.sample_best_subset import compute_gain_quadratic
from src.feature_extractor import get_all_features

RESULTS_DIR = Path("results")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

FONT_SIZES = {"small": 14, "medium": 18, "large": 24}

PLOT_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": FONT_SIZES.get("medium"),
    "axes.titlesize": FONT_SIZES.get("large"),
    "axes.labelsize": FONT_SIZES.get("large"),
    "xtick.labelsize": FONT_SIZES.get("large"),
    "ytick.labelsize": FONT_SIZES.get("large"),
    "legend.fontsize": FONT_SIZES.get("medium"),
    "figure.titlesize": FONT_SIZES.get("medium"),
    "text.usetex": True,
}

COLORS = {
    "pink": "#f0529c",
    "dark_teal": "#0a3235",
    "teal": "#105257",
    "purple": "#b11be8",
    "green": "#0fcb8c",
}


plt.rcParams.update(PLOT_PARAMS)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Plotting utilities", formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")

    # Define shared arguments
    shared_args = argparse.ArgumentParser(add_help=False)
    shared_args.add_argument("--output_path", type=Path, required=True, help="Path to save the PDF plot.")
    shared_args.add_argument("--figsize", type=int, nargs=2, default=[10, 10], help="Matplotlib figure size.")
    shared_args.add_argument("--random_seed", default=None, help="Set the random seed.")

    # Add new subcommand everytime you want to plot something new
    # In this way, we can centralize all plot customization into one script.
    parser_main_results = subparsers.add_parser("rewardbench_line", help="Plot main results line chart for RewardBench.", parents=[shared_args])
    parser_main_results.add_argument("--input_path", type=Path, required=False, help="Path to the results file.")

    parser_tag_heatmap = subparsers.add_parser("tag_heatmap", help="Plot heatmap of tag counts for a given dataset.", parents=[shared_args])
    parser_tag_heatmap.add_argument("--input_path", type=Path, required=False, help="Path to the results file.")

    parser_gain_distrib = subparsers.add_parser("gain_distrib", help="Plot the gain distribution for a dataset.", parents=[shared_args])
    parser_gain_distrib.add_argument("--dataset_path", action="append", help="Path to the dataset (dataset_name::path/to/features.jsonl).")
    parser_gain_distrib.add_argument("--model_path", type=Path, required=True, help="Path to the model.")

    parser_feat_distrib = subparsers.add_parser("feat_distrib", help="Plot a distribution of numerical (lexical features).", parents=[shared_args])
    parser_feat_distrib.add_argument("--dataset_path", action="append", help="Path to the dataset (dataset_name::path/to/features.jsonl).")
    parser_feat_distrib.add_argument("--feature", type=str, help="Feature (or field name) to plot.")
    parser_feat_distrib.add_argument("--feature_label", type=str, help="Feature (or field name) to use in xlabel.")

    parser_train_curve = subparsers.add_parser("train_curve", help="Plot a training curve for different models.", parents=[shared_args])
    parser_train_curve.add_argument("--curve", action="append", help="Train curve and its values (Linear::0.4324,0.6543,0.7888,0.8200).")

    parser_test_curve = subparsers.add_parser("test_curve", help="Plot a test curve from an input file.", parents=[shared_args])
    parser_test_curve.add_argument("--input_path", type=Path, required=False, help="Path to the results file.")

    parser_scaling =subparsers.add_parser("scaling", help="Plot a scaling chart", parents=[shared_args])
    parser_scaling.add_argument("--input_dir", type=Path, required=True, help="Path to the directory containing the scaling files (filename should be DIRECTORY/hs2p-SCALE-results-llama.csv)")
    parser_scaling.add_argument("--topk", type=int, default=1, help="Plot the top-k points per simulation.")

    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    if args.random_seed:
        logging.info(f"Setting the random seed to {args.random_seed}")
        random.seed(args.random_seed)

    cmd_map = {
        "rewardbench_line": plot_rewardbench_line,
        "tag_heatmap": plot_tag_heatmap,
        "gain_distrib": plot_gain_distrib,
        "feat_distrib": plot_feat_distrib,
        "train_curve": plot_train_curve,
        "test_curve": plot_test_curve,
        "scaling": plot_scaling_curve,
    }

    def _filter_args(func, kwargs):
        func_params = signature(func).parameters
        return {k: v for k, v in kwargs.items() if k in func_params}

    if args.command in cmd_map:
        plot_fn = cmd_map[args.command]
        kwargs = _filter_args(plot_fn, vars(args))
        plot_fn(**kwargs)
    else:
        logging.error(f"Unknown plotting command: {args.command}")


def plot_rewardbench_line(
    input_path: Path,
    output_path: Path,
    figsize: tuple[int, int],
):
    with input_path.open("r") as f:
        data = json.load(f)

    def plot(ax, dataset: str):
        levels = ["human_25", "human_50", "human_75"]

        random_avgs = [data[dataset][l]["random"]["score"] * 100 for l in levels]
        random_stds = [data[dataset][l]["random"]["std"] * 100 for l in levels]
        dm_avgs = [data[dataset][l]["datamodel_ours"]["score"] * 100 for l in levels]
        dm_stds = [data[dataset][l]["datamodel_ours"]["std"] * 100 for l in levels]

        # Add human_0 and human_100
        random_avgs.append(data[dataset]["human_100"]["score"] * 100)
        random_stds.append(data[dataset]["human_100"]["std"] * 100)
        dm_avgs.append(data[dataset]["human_100"]["score"] * 100)
        dm_stds.append(data[dataset]["human_100"]["std"] * 100)

        random_avgs.insert(0, data[dataset]["human_0"]["score"] * 100)
        random_stds.insert(0, data[dataset]["human_0"]["std"] * 100)
        dm_avgs.insert(0, data[dataset]["human_0"]["score"] * 100)
        dm_stds.insert(0, data[dataset]["human_0"]["std"] * 100)

        x_levels = ["$0\%$", "$25\%$", "$50\%$", "$75\%$", "$100\%$"]

        x = np.arange(len(x_levels))
        ax.errorbar(
            x,
            dm_avgs,
            yerr=dm_stds,
            label="Best Hybrid (Ours), Given Budget",
            marker="s",
            linestyle="-",
            linewidth=2,
            capsize=5,
            color=COLORS.get("teal"),
        )
        # Plot scores from random sampling
        ax.errorbar(
            x,
            random_avgs,
            yerr=random_stds,
            label="Random Hybrid, Given Budget",
            marker="o",
            linestyle="--",
            capsize=5,
            color=COLORS.get("pink"),
            alpha=0.5,
        )

        # Plot optimal scores
        # you need to precompute this: swaps / total_counts then interpolate
        x_opt = data[dataset]["optimal"].get("swaps")
        y_opt = data[dataset]["optimal"].get("score")
        opt_pct = data[dataset]["optimal"].get("swaps_pct")

        if x_opt and y_opt:
            y_opt = y_opt * 100
            # Plot optimal scores
            ax.plot(
                x_opt,
                y_opt,
                "*",
                markersize=20,
                color=COLORS.get("green"),
                # markeredgecolor=COLORS.get("dark_teal"),
                label="Best Hybrid (Ours), Unlimited Budget",
            )
            ax.text(
                x_opt + 0.75,
                y_opt + 0.75,
                f"{opt_pct:.1f}\%",
                fontsize=20,
                color=COLORS.get("green"),
                ha="center",
            )
            # Get the current y-axis limits
            ymin, ymax = ax.get_ylim()
            ax.vlines(
                x_opt,
                ymin=ymin,
                ymax=y_opt,
                color=COLORS.get("green"),
                linestyle="--",
            )
            ax.hlines(
                y_opt,
                xmin=0,
                xmax=x_opt,
                color=COLORS.get("green"),
                linestyle="--",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(x_levels)
        ax.set_xlabel("\% Direct Human Preference")
        ax.set_ylabel("RewardBench Score")
        ax.set_title(dataset, y=-0.50)
        ax.spines[["right", "top"]].set_visible(False)
        ax.yaxis.get_major_locator().set_params(integer=True)
        current_ylim = ax.get_ylim()
        ax.set_ylim([current_ylim[0], current_ylim[1] + 2])
        return ax

    fig, axs = plt.subplots(1, len(data), figsize=figsize)
    datasets = list(data.keys())
    for ax, dataset in zip(np.ravel(axs), datasets):
        plot(ax, dataset)
    # ax.legend(frameon=False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles[::-1],
        labels[::-1],
        frameon=False,
        fontsize=25,
        # I really don't want to parametrize the plotting
        # function too much. Manually update this so that
        # the legends fit nicely.
        loc="upper center",
        ncol=3,  # ncol = 3
        bbox_to_anchor=(0.5, 1.2),  # bbox_to_anchor= (0.5, 1.20)
    )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)


def plot_tag_heatmap(
    input_path: Path,
    output_path: Path,
    figsize: tuple[int, int],
):
    feats = get_all_features()
    df = pd.read_csv(input_path)[feats + ["Overall"]]

    columns_to_feature = {
        "rouge::min_val=0.33|max_val=0.67": "0.33$\leq$ROUGE-L$\leq$0.67",
        "rouge::min_val=0.67|max_val=1.0": "0.67$\leq$ROUGE-L$\leq$1.00",
        "token_len_diff::min_val=0.33|max_val=0.67": "0.33$\leq$Length diff. of responses$\leq$0.67",
        "token_len_diff::min_val=0.67|max_val=1.0": "0.67$\leq$Length diff of responses $\leq$1.00",
        "analyzer_closed_set::feature_name=subject_of_expertise|constraints=Computer sciences": "Subject of expertise: Computer sciences",
        "analyzer_closed_set::feature_name=subject_of_expertise|constraints=Chemistry": "Subject of expertise: Chemistry",
        "analyzer_scalar::feature_name=expertise_level|value=general public": "Expertise level: general public",
        "analyzer_scalar::feature_name=expertise_level|value=expert domain knowledge": "Expertise level: expert domain knowledge",
    }

    def group_list(lst, n):
        return [lst[i : i + n] for i in range(0, len(lst), n)]

    n = 16
    groups = group_list(list(columns_to_feature.keys()), 2)
    df = df.dropna().sample(n, random_state=42)
    fig, axs = plt.subplots(
        nrows=len(groups) + 1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [4, 4, 4, 4, 2]},
        # sharex=True,
    )
    cbar_ax = fig.add_axes([1.05, 0.3, 0.03, 0.4])
    for idx, (ax, group) in enumerate(zip(axs[:-1], groups)):
        feature_df = df[group + ["Overall"]].rename(columns=columns_to_feature)
        fmt = lambda x: f"{x/1000:.1f}k" if x >= 1000 else f"{int(x)}"
        input_data = feature_df.drop(columns=["Overall"]).transpose()
        sns.heatmap(
            input_data,
            ax=ax,
            annot=input_data.map(fmt),
            fmt="",
            cmap=colors.LinearSegmentedColormap.from_list(
                "custom_blue", ["#FFFFFF", COLORS.get("teal")]
            ),
            annot_kws={"size": 20},
            vmax=5000,
            vmin=0,
            cbar_ax=None if idx else cbar_ax,
            cbar=True if idx == 0 else False,
        )

        if idx == 0:
            # Only add labels in the first heatmap
            ax.set_xlabel(r"Candidate Datasets, $\{\hat{D}$\}", labelpad=20)
            ax.set_xticklabels([f"$\hat{{D}}_{{{i}}}$" for i in range(n)], rotation=0)
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()
            ax.tick_params(
                axis="x",
                which="both",
                length=0,
                labelbottom=False,
                labeltop=True,
            )
            colorbar = ax.collections[0].colorbar
            colorbar.set_label("Counts", labelpad=10)
            colorbar.ax.yaxis.set_label_position("left")
            colorbar.ax.yaxis.set_label_coords(0.5, 1.05)
            colorbar.ax.yaxis.label.set_rotation(0)
        else:
            ax.set_xticklabels([])
            ax.set_xticks([])

    score_df = df.copy(deep=True)
    score_df["Overall"] = df["Overall"] * 100

    sns.heatmap(
        score_df[["Overall"]].transpose(),
        ax=axs[-1],
        cmap=colors.LinearSegmentedColormap.from_list(
            "custom_blue", ["#FFFFFF", COLORS.get("pink")]
        ),
        cbar=False,
        annot=True,
        annot_kws={"size": 20},
        fmt=".1f",
    )
    axs[-1].set_xticks([])
    axs[-1].set_yticklabels([r"Perf$(\hat{R})$"], rotation=0, ha="right")
    axs[-1].set_xlabel("")
    axs[-1].set_ylabel("")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.parent / f"{output_path.stem}.svg", bbox_inches="tight")


def plot_gain_distrib(
    dataset_path: list[str],
    output_path: Path,
    model_path: Path,
    figsize: tuple[int, int] = (16, 4),
):
    model = joblib.load(model_path)
    feat_ext = (
        joblib.load(model_path.parent / "poly.pkl")
        if "quadratic" in str(model_path)
        else None
    )
    is_quadratic = True if feat_ext else False

    fig, axs = plt.subplots(1, len(dataset_path), figsize=figsize)
    for ax, dataset in zip(np.ravel(axs), dataset_path):
        dataset_name, dataset_fp = dataset.split("::")
        df = pd.read_json(dataset_fp, lines=True)
        if is_quadratic:
            df = compute_gain_quadratic(df, model, feat_ext)
        else:
            df = compute_gain_linear(df, model)

        sns.histplot(
            np.log1p(df["gain"] * 1e5),
            ax=ax,
            kde=True,
            stat="count",
            fill=True,
            bins=50,
            color="#105257",
        )
        ax.set_title(dataset_name)
        if not is_quadratic:
            ax.set_xlim([-0.01, 0.01])
        ax.set_xlim([-2, 2])
        ax.set_xlabel("Gain")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

        # Add black vertical line
        ax.axvline(x=0, color=COLORS.get("dark_teal"), linestyle="--", linewidth=5)
        for patch in ax.patches:
            if patch.get_x() < 0:
                patch.set_facecolor(COLORS.get("pink"))

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")


def plot_feat_distrib(
    dataset_path: list[str],
    output_path: Path,
    feature: str,
    feature_label: Optional[str] = None,
    figsize: tuple[int, int] = (16, 4),
):
    fig, axs = plt.subplots(1, 4, figsize=figsize)
    for ax, dataset in zip(np.ravel(axs), dataset_path):
        dataset_name, dataset_fp = dataset.split("::")
        df = pd.read_json(dataset_fp, lines=True)

        sns.histplot(
            df[feature],
            ax=ax,
            kde=True,
            stat="count",
            fill=True,
            bins=20,
            color="#105257",
        )
        ax.set_title(dataset_name)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if df[feature].max() <= 1:
            ax.set_xlim([0, 1])
        if feature == "len_longer":
            # Hacky
            feature_label = "Token length of\nlonger response"

        ax.set_xlabel(feature_label if feature_label else feature)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")


def plot_train_curve(
    output_path: Path,
    curve: list[str] = [],
    figsize: tuple[int, int] = (16, 4),
):

    if len(curve) == 0:
        raise ValueError("No value sent in '--curve'!")

    # Parse curve: Model::val_1,val_2,val_3,val_4
    fig, ax = plt.subplots()
    models = []
    values = []
    for c in curve:
        model, data = c.split("::")
        models.append(model)
        values.append([float(v) for v in data.split(",")])

    x = [25, 50, 75, 100]

    colors = [COLORS.get("pink"), COLORS.get("green"), COLORS.get("teal")]
    for model, vals, color in zip(models, values, colors):
        ax.plot(x, vals, marker="o", label=model, linewidth=2, color=color)

    ax.set_xlabel("Percentage of Training Data")
    ax.set_ylabel("Performance")
    ax.set_title("Training Curve for Different Models")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}%" for i in x])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.savefig(output_path, bbox_inches="tight")


def plot_test_curve(
    input_path: Path,
    output_path: Path,
    figsize: tuple[int, int] = (4, 4),
):
    df = (
        pd.read_csv(input_path)
        .sort_values(by="actual", ascending=True)
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=figsize)

    predicted = df["quadratic"] * 100  # scale same as others
    actual = df["actual"] * 100  # scale same as others
    ax.scatter(
        predicted,
        actual,
        marker="o",
        s=20,
        color=COLORS.get("teal"),
    )

    rmse = root_mean_squared_error(actual, predicted)
    ax.text(
        0.95,
        0.05,
        f"RMSE: {rmse:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=COLORS.get("pink"),
    )

    # # Add a diagonal line for reference (perfect prediction)
    min_val = min(predicted.min(), actual.min())
    max_val = max(predicted.max(), actual.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color=COLORS.get("pink"),
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_aspect("equal")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")


def plot_scaling_curve(
    input_dir: Path,
    topk: int,
    output_path: Path,
    figsize: tuple[int, int] = (12, 8),
):
    csv_files = list(input_dir.glob("*.csv"))
    logging.info(f"Found {len(csv_files)} files in {input_dir}")
    scaling_dict: dict[int, list[float]] = {
        int(file.stem.split("-")[1]): pd.read_csv(file)
        .sort_values(by="Overall", ascending=False)
        .head(topk)["Overall"]
        .to_list()
        for file in csv_files
    }

    fig, ax = plt.subplots(figsize=figsize)

    for scale, values in scaling_dict.items():
        x_values = [scale] * len(values)
        ax.scatter(
            x_values[1:],
            values[1:],
            label=f"Scale {scale}",
            s=40,
            color=COLORS.get("pink"),
            alpha=0.8,
        )
        ax.scatter(
            x=x_values[0],
            y=values[0],
            s=120,
            marker="*",
            color=COLORS.get("green"),
        )

    ax.set_xlabel(
        "Number of simulations given a fixed budget (67.7\%) on Helpsteer2-Preferences"
    )
    ax.set_ylabel("Overall Score")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xticks([256, 512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels([str(x) for x in [256, 512, 1024, 2048, 4096, 8192]])

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
