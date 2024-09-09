import argparse
import pandas as pd
from pathlib import Path
import tqdm


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--results_path", type=Path, help="Path to the 'experiments.csv' file obtained from running fetch_evals_rewardbench.py")
    parser.add_argument("--mixes_dir", type=Path, help="Directory containing all the JSONL subset files.")
    parser.add_argument("--feats_dataset_id", type=str, default="01J7C755BWNJGA0ZEB9NM5D12P", help="Beaker ID containing the extracted lexical features and metadata features.")
    parser.add_argument("--dataset")
    # fmt: on
    pass


def main():
    pass


if __name__ == "__main__":
    main()
