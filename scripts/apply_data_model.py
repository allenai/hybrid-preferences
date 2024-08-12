import argparse
from pathlib import Path

import sys
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
    # fmt: off
    description = """Apply data model to get swapped preferences.

This CLI expects a JSONL file with fields `pref_human` and `pref_gpt4`.
Then, it will output a JSONL file containing the following fields:
- `pref`: the final preference used for reward model training
- `is_swapped`: whether that instance fulfilled the features passed.
- `features_used`: a comma-separated string of features used for this instance.

You can select a number of features by passing arguments to the `--features` option.
By default all features will be computed.
"""

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input_path", type=Path, required=True, help="Path to the JSONL file containing preferences.")
    parser.add_argument("--output_path", type=Path, required=True, help="Path to save the output JSONL file.")
    parser.add_argument("--features", nargs="*", default=None, help="Features to include. To show all available features, pass --show_all_features.")
    parser.add_argument("--show_all_features", action="store_true", default=False, help="If set, will just show all available features and exit the CLI.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    if args.show_all_features:
        logging.info("Hi")


if __name__ == "__main__":
    main()
