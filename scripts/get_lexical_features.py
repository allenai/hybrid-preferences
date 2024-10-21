import argparse
import sys
import logging
from pathlib import Path

import pandas as pd

from src.feature_extractor import FeatureExtractor

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
    # fmt: off
    description = "Compute lexical features given a dataset with prompts and responses."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input_path", type=Path, required=True, help="Path to a JSONL file containing prompts and completions.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the extracted features (individual JSONL files).")
    parser.add_argument("--features", nargs="*", default=None, help="Features to include. To show all available features. If not set, will try all feature combinations.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    df = pd.read_json(args.input_path, lines=True)

    if "prompt_hash" not in df.columns:
        raise ValueError(
            f"Column {prompt_hash} required. Must contain a unique ID for each prompt."
        )
    if "response_a" not in df.columns or "response_b" not in df.columns:
        raise ValueError(
            f"Response columns should be 'response_a' and 'response_b' respectively."
        )
    if "text" not in df.columns:
        raise ValueError(f"Prompt column should be 'text'")

    extractor = FeatureExtractor(
        df,
        id_col="prompt_hash",
        prompt_col="text",
        completion_a_col="response_a",
        completion_b_col="response_b",
        keep_features=args.output_dir,
    )

    logging.info("Extracting features")
    extractor(features=features, threshold=1.0)


if __name__ == "__main__":
    main()
