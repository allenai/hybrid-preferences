import argparse
import hashlib
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from src.feature_extractor import FeatureExtractor, sample_feature_combinations

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
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
    entity_sim::threshold=0.95|model_name=en_core_web_lg|n_process=4

You can use the `--show_all_features` flag to get a list of all available
features.  If you don't pass anything in the `--features` option, this CLI will
extract all features.

"""

    # fmt: off
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")

    # Define shared arguments
    shared_args = argparse.ArgumentParser(add_help=False)
    shared_args.add_argument("--input_path", type=Path, required=False, help="Path to the JSONL file containing preferences.")
    shared_args.add_argument("--output_dir", type=Path, required=False, help="Directory to save the output JSONL file.")
    shared_args.add_argument("--num_instances", type=int, default=7000, help="Number of instances to save in the output file.")
    shared_args.add_argument("--threshold", type=float, default=1.0, help="Percentage of total features to be active in order to swap w/ human preferences.")
    shared_args.add_argument("--keep_features_dir", type=Path, default=None, help="If set, will store all collected features in this directory.")
    shared_args.add_argument("--append_to_experiments_file", type=Path, default=None, help="If set, will append to an experiments TXT file to be used for submitting TPU training jobs.")
    shared_args.add_argument("--min_num_swaps", type=int, default=1000, help="If set, will not save combinations that have human swaps less than this number.")
    shared_args.add_argument("--random_seed", type=int, default=None, help="Set the random seed.")

    # Arguments for 'single' command
    parser_single = subparsers.add_parser("single", help="Run single data model run", parents=[shared_args])
    parser_single.add_argument("--features", nargs="*", default=None, help="Features to include. To show all available features. If not set, will try all feature combinations. Pass --show_all_combinations to show all features.")
    parser_single.add_argument("--show_all_features", action="store_true", help="Show all available features and exit.")

    # Arguments for 'multi' command
    parser_multi = subparsers.add_parser("multi", help="Run multiple combinations", parents=[shared_args])
    parser_multi.add_argument("--n_sample_combinations", type=int, default=None, help="Sample the combinations to run.")
    parser_multi.add_argument("--ignore_list", nargs="*", default=None, help="Ignore combinations that include these features.")

    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    all_features, all_combinations = sample_feature_combinations(
        n_bins=3,
        meta_analyzer_n_samples=700,
    )

    df = pd.read_json(args.input_path, lines=True)
    if not {"pref_human", "pref_gpt4"}.issubset(set(list(df.columns))):
        logging.error("Columns 'pref_human' and 'pref_gpt4' should be present!")
        sys.exit(1)

    feature_extractor = FeatureExtractor(
        df,
        id_col="prompt_hash",
        prompt_col="text",
        completion_a_col="response_a",
        completion_b_col="response_b",
        keep_features=args.keep_features_dir,
    )

    if args.command == "single":
        if args.show_all_features:
            logging.info(f"Available features: {', '.join(all_features)}")
            sys.exit(0)

        apply_data_model(
            feature_extractor,
            output_dir=args.output_dir,
            features=args.features,
            threshold=args.threshold,
            random_seed=args.random_seed,
            num_instances=args.num_instances,
            append_to_experiments_file=args.append_to_experiments_file,
            min_num_swaps=args.min_num_swaps,
        )
    elif args.command == "multi":

        feats_to_run = (
            random.sample(
                all_combinations, min(args.n_sample_combinations, len(all_combinations))
            )
            if args.n_sample_combinations
            else all_combinations
        )

        if args.ignore_list:
            logging.info(
                f"Ignoring feature combinations that include these features: {', '.join(args.ignore_list)}"
            )
            feats_to_run = filter_ignored_features(
                feats_to_run, ignore_values=args.ignore_list
            )
            logging.info(f"Updated number of feature combinations: {len(feats_to_run)}")

        logging.info(
            f"There are {len(all_combinations)} total feature combinations."
            f" Running {len(feats_to_run)}."
        )

        for idx, feature_combination in enumerate(feats_to_run):
            logging.info(
                f"*** Feature {idx+1}/{len(feats_to_run)}: {', '.join(feature_combination)} ***"
            )

            apply_data_model(
                feature_extractor,
                output_dir=args.output_dir,
                features=feature_combination,
                threshold=args.threshold,
                random_seed=args.random_seed,
                num_instances=args.num_instances,
                append_to_experiments_file=args.append_to_experiments_file,
                min_num_swaps=args.min_num_swaps,
            )


def apply_data_model(
    extractor: "FeatureExtractor",
    *,
    output_dir: Path,
    features: Optional[list[str]] = None,
    threshold: float = 1.0,
    random_seed: Optional[int] = None,
    num_instances: int = 7000,
    append_to_experiments_file: Optional[Path] = None,
    min_num_swaps: int = 0,
):
    if random_seed:
        logging.info(f"Setting random seed to {random_seed}")
        random.seed(random_seed)
    if not features:
        logging.info(
            "Will extract all available features using their default parameters"
        )
        features = list(extractor.REGISTERED_EXTRACTORS.keys())

    logging.info("Extracting features")
    extracted_df = extractor(features=features, threshold=threshold)

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

    # Sample converted instances
    if num_instances < len(converted_annotations):
        converted_annotations = random.sample(converted_annotations, num_instances)
        logging.info(f"Sampled {num_instances} instances from the total.")

    output_dir.mkdir(parents=True, exist_ok=True)

    feature_tags = []
    for feature in features:
        if "analyzer" in feature:
            # Cleanup the names
            _, feature_params = feature.split("::")
            feature_name, feature_value = feature_params.split("|")
            _, name = feature_name.split("=")
            _, value = feature_value.split("=")
            value = value.replace(" ", "_")
            feature_tags.append(f"{name}={value}")
        else:
            feature_tags.append(feature)
    tag = "___".join(feature_tags).replace("::", "__").replace("=", "-")
    feats_id = hashlib.md5(tag.encode("utf-8")).hexdigest()
    num_swaps = 0
    for annotation in converted_annotations:
        if annotation.get("is_swapped"):
            num_swaps += 1

    if num_swaps > min_num_swaps:
        output_path = (
            output_dir
            / f"human_datamodel_{num_instances}_FEATS_{feats_id}_SWAPS_{num_swaps}.jsonl"
        )
        with output_path.open("w") as f:
            for annotation in converted_annotations:
                f.write(json.dumps(annotation) + "\n")
        logging.info(f"Saved to {output_path}")

        # Create a TXT file so that it's easier to submit tpu jobs
        experiments_file: Path = append_to_experiments_file
        if experiments_file:
            experiment_name = output_path.stem
            if experiments_file.exists():
                with experiments_file.open("r") as f:
                    data = f.read().splitlines()
                existing_hashes = [extract_hash(d) for d in data]
                if feats_id in existing_hashes:
                    logging.info(f"Tag set '{tag}' ({feats_id}) already exists! ")
                else:
                    logging.info(f"Appending {experiment_name} to {experiments_file}")
                    with experiments_file.open("a") as f:
                        f.write("\n" + f"{experiment_name}::{tag}")
            else:
                logging.info(
                    f"{experiments_file} not found. Generating new and appending experiment {experiment_name}"
                )
                with experiments_file.open("w") as f:
                    f.write(f"{experiment_name}::{tag}")
    else:
        logging.info(
            f"Tag set '{tag}' resulted into {num_swaps} swaps (< {min_num_swaps})! Skipping"
        )


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
        "id": instance["id"],
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


def filter_ignored_features(
    list_of_features: list[list[str]], ignore_values: list[str]
) -> list[list[str]]:
    def contains_ignore_value(feature_list: list[str]):
        return any(
            ignore in feature for feature in feature_list for ignore in ignore_values
        )

    filtered_lists = [
        feature_list
        for feature_list in list_of_features
        if not contains_ignore_value(feature_list)
    ]
    return filtered_lists


def extract_hash(string):
    match = re.search(r"FEATS_(.*?)_SWAPS", string)
    return match.group(1) if match else None


if __name__ == "__main__":
    main()
