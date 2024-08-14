import argparse
import json
import logging
import random
import sys
from pathlib import Path

import pandas as pd

from src.feature_extractor import FeatureExtractor
from src.feature_extractor import get_all_feature_combinations

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
    # fmt: off
    description = """Apply data model to get swapped preferences.

This CLI expects a JSONL file with fields `pref_human` and `pref_gpt4`.  Then,
it will output a JSONL file in the provided directory containing the following
fields:

- `pref`: the final preference used for reward model training
- `is_swapped`: whether that instance fulfilled the features passed.
- `features_used`: a comma-separated string of features used for this instance.

You can select a number of features by passing arguments to the `--features`
option.  All features will be computed by default.  Some features can be
parametrized. You can do so by first appending a double colon (::), and then
passing the parameters as a name=value dictionary. Remember that booleans should
be 0 or 1.

For example:

    [feature_name]::[param1]=[value1],[param2]=[value2]
    entity_sim::threshold=0.95,model_name=en_core_web_lg,n_process=4

You can use the `--show_all_features` flag to get a list of all available
features.  If you don't pass anything in the `--features` option, this CLI will
extract all features.

"""

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input_path", type=Path, required=False, help="Path to the JSONL file containing preferences.")
    parser.add_argument("--output_dir", type=Path, required=False, help="Directory to save the output JSONL file.")
    parser.add_argument("--num_instances", type=int, default=7000, help="Number of instances to save in the output file.")
    parser.add_argument("--features", nargs="*", default=None, help="Features to include. To show all available features. If not set, will try all feature combinations. Pass --show_all_combinations to show all features.")
    parser.add_argument("--threshold", type=float, default=1.0, help="Percentage of total features to be active in order to swap w/ human preferences.")
    parser.add_argument("--keep_features_dir", type=Path, default=None, help="If set, will store all collected features in this directory.")
    parser.add_argument("--append_to_experiments_file", type=Path, default=None, help="If set, will append to an experiments TXT file to be used for submitting TPU training jobs.")
    parser.add_argument("--random_seed", type=int, default=42, help="Set the random seed.")
    parser.add_argument("--show_all_features", action="store_true", help="Show all available features and exit.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    all_features, _ = get_all_feature_combinations()
    if args.show_all_features:
        logging.info(f"Available features: {', '.join(all_features)}")
        sys.exit(0)

    random.seed(args.random_seed)
    df = pd.read_json(args.input_path, lines=True)
    if not {"pref_human", "pref_gpt4"}.issubset(set(list(df.columns))):
        logging.error("Columns 'pref_human' and 'pref_gpt4' should be present!")
        sys.exit(1)

    # Swap preferences
    logging.info("Swapping preferences")
    extractor = FeatureExtractor(
        df,
        id_col="prompt_hash",
        prompt_col="text",
        completion_a_col="response_a",
        completion_b_col="response_b",
        keep_features=args.keep_features_dir,
    )
    features = args.features
    if not features:
        logging.info(
            "Will extract all available features using their default parameters"
        )
        features = list(extractor.REGISTERED_EXTRACTORS.keys())
    extracted_df = extractor(features=features, threshold=args.threshold)

    # Convert to DPO training format
    logging.info("Converting annotations into DPO training format")
    annotations = extracted_df.to_dict(orient="records")
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
    tag = "___".join(features).replace("::", "__").replace("=", "-")
    output_path = output_dir / f"human_datamodel_{args.num_instances}_FEATS_{tag}.jsonl"
    with output_path.open("w") as f:
        for annotation in converted_annotations:
            f.write(json.dumps(annotation) + "\n")
    logging.info(f"Saved to {output_path}")

    experiments_file: Path = args.append_to_experiments_file
    if experiments_file:
        experiment_name = output_path.stem
        if experiments_file.exists():
            logging.info(
                f"Appending experiment {experiment_name} to {experiments_file}"
            )
            with experiments_file.open("a") as f:
                f.write("\n" + experiment_name)
        else:
            logging.info(
                f"{experiments_file} not found. Generating new and appending experiment {experiment_name}"
            )
            with experiments_file.open("w") as f:
                f.write(experiment_name)


def convert_to_dpo_format(
    instance: dict[str, str], preference_label: str
) -> dict[str, str]:
    conversation_a = [
        {"content": instance["prompt"], "role": "user"},
        {"content": instance["completion_a"], "role": "assistant"},
    ]
    conversation_b = [
        {"content": instance["prompt"], "role": "user"},
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
        "prompt": instance["prompt"],
        "chosen": chosen,
        "chosen_model": chosen_model,
        "rejected": rejected,
        "rejected_model": rejected_model,
        "features_used": instance.get("features_used"),
        "is_swapped": instance.get("is_swapped"),
    }


if __name__ == "__main__":
    main()
