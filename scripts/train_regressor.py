import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import tqdm

from beaker import Beaker, Experiment
from scripts.fetch_evals_rewardbench import fetch_evals_rewardbench as fetch_results

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input_mixes_dir", type=Path, help="Directory containing all the JSONL subset files.")
    parser.add_argument("--feats_dataset_id", type=str, default="01J7C755BWNJGA0ZEB9NM5D12P", help="Beaker ID containing the extracted lexical features and metadata features.")
    parser.add_argument("--beaker_workspace", default="ai2/ljm-oe-adapt", help="Beaker workspace to fetch experiments.")
    parser.add_argument("--experiment_prefix", default="rm-eval-", help="Prefix for experiments to fetch.")
    parser.add_argument("--experiments_file", default=None, type=Path, help="Path to a TXT file containing a list that maps an experiment to the features.")
    # fmt: on
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

    results_df = fetch_results(
        beaker=beaker,
        beaker_workspace=args.beaker_workspace,
        experiment_prefix=args.experiment_prefix,
        experiments_file=args.experiments_file,
    )


if __name__ == "__main__":
    main()
