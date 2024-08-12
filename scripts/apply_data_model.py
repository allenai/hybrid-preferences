import json
import argparse
import random
from pathlib import Path
import sys
import logging

import pandas as pd

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
Then, it will output a JSONL file in the provided directory containing the following fields:
- `pref`: the final preference used for reward model training
- `is_swapped`: whether that instance fulfilled the features passed.
- `features_used`: a comma-separated string of features used for this instance.

You can select a number of features by passing arguments to the `--features` option.
All features will be computed by default.
"""

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input_path", type=Path, required=True, help="Path to the JSONL file containing preferences.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the output JSONL file.")
    parser.add_argument("--num_instances", type=int, default=7000, help="Number of instances to save in the output file.")
    parser.add_argument("--features", nargs="*", default=None, help="Features to include. To show all available features, pass --show_all_features.")
    parser.add_argument("--show_all_features", action="store_true", default=False, help="If set, will just show all available features and exit the CLI.")
    parser.add_argument("--random_seed", type=int, default=42, help="Set the random seed.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    random.seed(args.random_seed)

    if args.show_all_features:
        logging.info("Features you can use")

    df = pd.read_json(args.input_path, lines=True)
    if not {"pref_human", "pref_gpt4"}.issubset(set(list(df.columns))):
        logging.error("Columns 'pref_human' and 'pref_gpt4' should be present!")
        sys.exit(1)

    # Swap preferences
    logging.info("Swapping preferences")

    # Convert to DPO training format
    logging.info("Converting annotations into DPO training format")
    annotations = df.to_dict(orient="records")
    converted_annotations = []
    for annotation in annotations:
        if "model_a" not in annotation:
            annotation["model_a"] = ""
        if "model_b" not in annotation:
            annotation["model_b"] = ""
        if "source" not in annotation:
            annotation["source"] = ""
        if "highest_level_degree" not in annotation:
            annotation["highest_level_degree"] = ""
        assert "pref" in annotation, "Missing 'pref' key in instance."
        converted_instance = convert_to_dpo_format(annotation, annotation["pref"])
        if converted_instance is not None:
            converted_annotations.append(converted_instance)
    logging.info(f"Number of instances after selection: {len(converted_annotations)}")

    if args.num_instances < len(converted_annotations):
        converted_annotations = random.sample(converted_annotations, args.num_instances)
        logging.info(f"Sampled {args.num_instances} instances from the total.")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_dir.open("w") as f:
        for annotation in converted_annotations:
            f.write(json.dumps(annotation) + "\n")


def convert_to_dpo_format(
    instance: dict[str, str], preference_label: str
) -> dict[str, str]:
    conversation_a = [
        {"content": instance["text"], "role": "user"},
        {"content": instance["completion_a"], "role": "assistant"},
    ]
    conversation_b = [
        {"content": instance["text"], "role": "user"},
        {"content": instance["completion_b"], "role": "assistant"},
    ]
    if preference_label.lower() in [
        "a-is-slightly-better",
        "a-is-clearly-better",
        "a-is-better",
    ]:
        chosen = conversation_a
        chosen_model = instance["model_a"]
        rejected = conversation_b
        rejected_model = instance["model_b"]
    elif preference_label.lower() in [
        "b-is-slightly-better",
        "b-is-clearly-better",
        "b-is-better",
    ]:
        chosen = conversation_b
        chosen_model = instance["model_b"]
        rejected = conversation_a
        rejected_model = instance["model_a"]
    elif preference_label.lower() == "tie":
        return None
    else:
        raise ValueError(f"Invalid preference label: {preference_label}")
    return {
        "source": instance["source"],
        "highest_level_degree": instance["highest_level_degree"],
        "prompt": instance["text"],
        "chosen": chosen,
        "chosen_model": chosen_model,
        "rejected": rejected,
        "rejected_model": rejected_model,
        "features_used": instance.get("features_used"),
        "is_swapped": instance.get("is_swapped"),
    }


if __name__ == "__main__":
    main()
