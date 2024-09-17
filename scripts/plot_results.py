import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
import numpy as np
import logging
from inspect import signature

RESULTS_DIR = Path("results")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

FONT_SIZES = {"small": 14, "medium": 18, "large": 24}

PLOT_PARAMS = {
    "font.family": "Times New Roman",
    "font.size": FONT_SIZES.get("medium"),
    "axes.titlesize": FONT_SIZES.get("large"),
    "axes.labelsize": FONT_SIZES.get("large"),
    "xtick.labelsize": FONT_SIZES.get("large"),
    "ytick.labelsize": FONT_SIZES.get("large"),
    "legend.fontsize": FONT_SIZES.get("medium"),
    "figure.titlesize": FONT_SIZES.get("medium"),
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
    shared_args.add_argument("--figsize", type=int, nargs=2, default=[12, 12], help="Path to save the PDF plot.")

    # Add new subcommand everytime you want to plot something new
    # In this way, we can centralize all plot customization into one script.
    parser_main_results = subparsers.add_parser("rewardbench_line", help="Plot main results line chart for RewardBench", parents=[shared_args])

    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    cmd_map = {"rewardbench_line": plot_rewardbench_line}

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

        random_avgs = [data[dataset][l]["random"]["score"] for l in levels]
        random_stds = [data[dataset][l]["random"]["std"] for l in levels]
        topk_avgs = [data[dataset][l]["top_k_gain_ours"]["score"] for l in levels]
        topk_stds = [data[dataset][l]["top_k_gain_ours"]["std"] for l in levels]

        # Add human_0 and human_100
        random_avgs.append(data[dataset]["human_100"]["score"])
        random_stds.append(data[dataset]["human_100"]["std"])
        topk_avgs.append(data[dataset]["human_100"]["score"])
        topk_stds.append(data[dataset]["human_100"]["std"])

        random_avgs.insert(0, data[dataset]["human_0"]["score"])
        random_stds.insert(0, data[dataset]["human_0"]["std"])
        topk_avgs.insert(0, data[dataset]["human_0"]["score"])
        topk_stds.insert(0, data[dataset]["human_0"]["std"])

        x_levels = ["0%", "25%", "50%", "75%", "100%"]

        x = np.arange(len(x_levels))
        ax.errorbar(
            x,
            random_avgs,
            yerr=random_stds,
            label="Random",
            marker="o",
            linestyle="-",
            capsize=5,
        )
        # Plot Top-k Gain (Ours) scores
        ax.errorbar(
            x,
            topk_avgs,
            yerr=topk_stds,
            label="Top-k Gain (Ours)",
            marker="s",
            linestyle="--",
            capsize=5,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(x_levels)
        ax.set_xlabel("Pct. Direct Human Preference")
        ax.set_ylabel("RewardBench Score")
        ax.set_title(dataset)
        # ax.set_ylim([0.5, 1])
        ax.legend()

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    datasets = list(data.keys())
    for ax, dataset in zip(np.ravel(axs), datasets):
        plot(ax, dataset)

    plt.tight_layout()
    fig.savefig(output_path)


if __name__ == "__main__":
    main()
