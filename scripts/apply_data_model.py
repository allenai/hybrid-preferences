import argparse
from pathlib import Path


def get_args():
    # fmt: off
    description = """Apply data model to get swapped preferences.

This CLI expects a JSONL file with fields `pref_human` and `pref_gpt4`.
Then, it will output a JSONL file containing the following fields:
- `pref`: the final preference used for reward model training
- `is_swapped`: whether that instance fulfilled the features passed.
- `features_used`: a comma-separated string of features used for this instance.

You can select a number of features by passing arguments to the `--features` option.
If None is passed, then all features will be computed.
"""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input_path", type=Path, required=True, help="Path to the JSONL file containing preferences.")
    parser.add_argument("--output_path", type=Path, required=True, help="Path to save the output JSONL file.")
    parser.add_argument("--features", nargs="*", default=None, help="Features to include. To show all available features, pass --show_all_features.")
    # fmt: on
    return parser.parse_args()


def main():
    pass


if __name__ == "__main__":
    main()
