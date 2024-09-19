import argparse
import random
import json
import logging
import sys
from inspect import signature
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
import seaborn as sns

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
    shared_args.add_argument("--input_path", type=Path, required=True, help="Path to the results file.")
    shared_args.add_argument("--output_path", type=Path, required=True, help="Path to save the PDF plot.")
    shared_args.add_argument("--figsize", type=int, nargs=2, default=[10, 10], help="Matplotlib figure size.")
    shared_args.add_argument("--random_seed", default=None, help="Set the random seed.")

    # Add new subcommand everytime you want to plot something new
    # In this way, we can centralize all plot customization into one script.
    parser_main_results = subparsers.add_parser("rewardbench_line", help="Plot main results line chart for RewardBench.", parents=[shared_args])
    parser_tag_heatmap = subparsers.add_parser("tag_heatmap", help="Plot heatmap of tag counts for a given dataset.", parents=[shared_args])

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
    input_path: Path, output_path: Path, figsize: tuple[int, int]
):
    with input_path.open("r") as f:
        data = json.load(f)

    def plot(ax, dataset: str):
        levels = ["human_25", "human_50", "human_75"]

        random_avgs = [data[dataset][l]["random"]["score"] * 100 for l in levels]
        random_stds = [data[dataset][l]["random"]["std"] * 100 for l in levels]
        topk_avgs = [data[dataset][l]["top_k_gain_ours"]["score"] * 100 for l in levels]
        topk_stds = [data[dataset][l]["top_k_gain_ours"]["std"] * 100 for l in levels]

        # Add human_0 and human_100
        random_avgs.append(data[dataset]["human_100"]["score"] * 100)
        random_stds.append(data[dataset]["human_100"]["std"] * 100)
        topk_avgs.append(data[dataset]["human_100"]["score"] * 100)
        topk_stds.append(data[dataset]["human_100"]["std"] * 100)

        random_avgs.insert(0, data[dataset]["human_0"]["score"] * 100)
        random_stds.insert(0, data[dataset]["human_0"]["std"] * 100)
        topk_avgs.insert(0, data[dataset]["human_0"]["score"] * 100)
        topk_stds.insert(0, data[dataset]["human_0"]["std"] * 100)

        x_levels = ["$0\%$", "$25\%$", "$50\%$", "$75\%$", "$100\%$"]

        x = np.arange(len(x_levels))
        # Plot Top-k Gain (Ours) scores
        ax.errorbar(
            x,
            topk_avgs,
            yerr=topk_stds,
            label="Top-k Gain (Ours)",
            marker="s",
            linestyle="-",
            linewidth=2,
            capsize=5,
            color=COLORS.get("dark_teal"),
        )
        # Plot scores from random sampling
        ax.errorbar(
            x,
            random_avgs,
            yerr=random_stds,
            label="Random",
            marker="o",
            linestyle="--",
            capsize=5,
            color=COLORS.get("pink"),
            alpha=0.5,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(x_levels)
        ax.set_xlabel("\% Direct Human Preference")
        ax.set_ylabel("RewardBench Score")
        ax.set_title(dataset)
        ax.spines[["right", "top"]].set_visible(False)
        ax.yaxis.get_major_locator().set_params(integer=True)
        # ax.set_ylim([0.5, 1])
        return ax

    fig, axs = plt.subplots(1, 4, figsize=figsize)
    datasets = list(data.keys())
    for ax, dataset in zip(np.ravel(axs), datasets):
        plot(ax, dataset)
    # ax.legend(frameon=False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        frameon=False,
        ncol=2,
        bbox_to_anchor=(0.5, -0.10),
    )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")


def plot_tag_heatmap(
    input_path: Path,
    output_path: Path,
    figsize: tuple[int, int],
):
    feats = get_all_features()
    df = pd.read_csv(input_path)[feats + ["Overall"]]

    columns_to_feature = {
        "bertscore::min_val=0.67|max_val=1.0": "0.67$\leq$BERTScore$\leq$1.00",
        "token_len_diff::min_val=0.33|max_val=0.67": "0.33$\leq$Diff. token length$\leq$0.67",
        "analyzer_closed_set::feature_name=subject_of_expertise|constraints=Computer sciences": "Subject of expertise: Computer sciences",
        "analyzer_scalar::feature_name=safety_concern|value=safe": "Safety concern: safe",
        "analyzer_scalar::feature_name=complexity_of_intents|value=simple": "Complexity of intent: simple",
        "analyzer_scalar::feature_name=expertise_level|value=general public": "Expertise level: general public",
        "analyzer_scalar::feature_name=expertise_level|value=expert domain knowledge": "Expertise level: expert domain knowledge",
    }
    df = df[list(columns_to_feature.keys()) + ["Overall"]].rename(
        columns=columns_to_feature
    )

    # Rename columns:
    df = (
        df.dropna().head(200)
        # .rename(columns={col: f"t{idx}" for idx, col in enumerate(df.columns)})
    )
    # Normalize
    # df = (df - df.mean()) / df.std()
    n = 16
    df = df.sample(n, random_state=42)

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        figsize=figsize,
        gridspec_kw={"height_ratios": [4, 1]},
        # sharex=True,
    )
    sns.heatmap(
        df.drop(columns=["Overall"]).transpose(),
        ax=ax1,
        annot=False,
        cmap=colors.LinearSegmentedColormap.from_list(
            "custom_blue", ["#FFFFFF", COLORS.get("dark_teal")]
        ),
    )
    ax1.set_xlabel(r"Proxy Dataset, $\hat{D}$", labelpad=20)
    ax1.set_xticklabels([f"$\hat{{d}}_{{{i}}}$" for i in range(n)], rotation=0)
    ax1.xaxis.set_label_position("top")
    ax1.xaxis.tick_top()
    ax1.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        length=0,
        labelbottom=False,
        labeltop=True,
    )
    colorbar = ax1.collections[0].colorbar
    colorbar.set_label("Counts", labelpad=10)
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.ax.yaxis.set_label_coords(-0.5, 1.05)
    colorbar.ax.yaxis.label.set_rotation(0)

    sns.heatmap(
        df[["Overall"]].transpose(),
        ax=ax2,
        cmap=colors.LinearSegmentedColormap.from_list(
            "custom_blue", ["#FFFFFF", COLORS.get("pink")]
        ),
        cbar=True,
        annot=True,
        fmt=".2f",
    )
    ax2.set_xticks([])
    ax2.set_yticklabels([r"Perf$(\hat{R})$"], rotation=0, ha="right")
    # ax2.set_yticks(["Perf(R)"])
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    # Trick to make it look aligned without showing the colorbar
    colorbar = ax2.collections[0].colorbar
    colorbar.ax.set_visible(False)
    colorbar.outline.set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
