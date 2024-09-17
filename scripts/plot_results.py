import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging
from inspect import signature

RESULTS_DIR = Path("results")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Plotting utilities", formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")

    # Define shared arguments
    shared_args = argparse.ArgumentParser(add_help=False)
    shared_args.add_argument("--input_path", type=Path, required=True, help="Path to the results file.")
    shared_args.add_argument("--output_path", type=Path, required=True, help="Path to save the PDF plot.")

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


def plot_rewardbench_line(input_path: Path, output_path: Path):

    breakpoint()


if __name__ == "__main__":
    main()
