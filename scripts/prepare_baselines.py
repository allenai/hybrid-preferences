import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
    # fmt: off
    description = "Get baseline datasets and their respective experiments.txt file"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output_dir", type=Path, help="Directory to save the JSONL files and the TXT experiments file.")
    parser.add_argument("--input_path", type=Path, help="Dataset path to create baselines on.")
    parser.add_argument("--prefix", type=str, help="Prefix to add to the output files.")
    parser.add_argument("--id_col", type=str, default="id", help="Column that contains the unique ID for each instance.")
    parser.add_argument("--prompt_col", type=str, default="text", help="Column that contains the text.")
    parser.add_argument("--completion_a_col", type=str, default="response_a", help="Column that contains response A.")
    parser.add_argument("--completion_b_col", type=str, default="response_b", help="Column that contains response B.")
    parser.add_argument("--num_instances", type=int, default=7000, help="Number of instances to sample.")
    parser.add_argument("--random_seed", type=int, default=42, help="Set random seed.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    logging.info(f"Setting random seed to {args.random_seed}")
    random.seed(args.random_seed)

    annotation_df = pd.read_json(args.input_path, lines=True)
    assert "pref_human" in annotation_df.columns, "Must contain 'pref_human' column!"
    assert "pref_gpt4" in annotation_df.columns, "Must contain 'pref_gpt4' column!"

    # Normalize column names
    annotation_df["id"] = annotation_df[args.id_col]
    annotation_df["prompt"] = annotation_df[args.prompt_col]
    annotation_df["completion_a"] = annotation_df[args.completion_a_col]
    annotation_df["completion_b"] = annotation_df[args.completion_b_col]

    def swap_prefs(df, r: float):
        _df = df.copy(deep=True)
        _df["is_swapped"] = np.random.rand(len(_df)) < r
        _df["pref"] = np.where(_df["is_swapped"], _df["pref_human"], _df["pref_gpt4"])
        return _df

    baselines = {
        "human": swap_prefs(annotation_df, r=1.0),
        "human_75": swap_prefs(annotation_df, r=0.75),
        "human_50": swap_prefs(annotation_df, r=0.50),
        "human_25": swap_prefs(annotation_df, r=0.25),
        "gpt4": swap_prefs(annotation_df, r=0),
        "random": swap_prefs(annotation_df, r=0.50),
    }

    experiments = []
    for baseline, annotation_df in baselines.items():
        annotations = annotation_df.to_dict(orient="records")
        converted_instances = get_converted_instances(annotations, args.num_instances)
        num_swaps = 0
        for instance in converted_instances:
            if instance.get("is_swapped"):
                num_swaps += 1

        pct_swaps = (num_swaps / len(converted_instances)) * 100
        logging.info(f"Baseline '{baseline}' has {num_swaps} ({pct_swaps:.2f}) swaps!")
        experiment_name = (
            f"{args.prefix}_{baseline}_SWAPS_{num_swaps}_SEED_{args.random_seed}"
        )
        experiments.append(experiment_name)
        output_path: Path = args.output_dir / f"{experiment_name}.jsonl"
        with output_path.open("w") as f:
            for instance in converted_instances:
                f.write(json.dumps(instance) + "\n")
        logging.info(f"Saved to {output_path}")

    experiments_file: Path = (
        args.output_dir / f"{args.prefix}-experiments-SEED-{args.random_seed}.txt"
    )
    experiments_file.write_text("\n".join(experiments))
    logging.info(f"Saved experiments to {experiments_file}")
    logging.info(f"Upload the JSONL files to GCS under the {args.prefix}/ directory")
    logging.info("And then, run the `scripts/submit_tpu_train_job.py` file.")


def get_converted_instances(
    annotations: list[dict[str, str]], num_instances: int
) -> list[dict[str, str]]:
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
        assert "id" in annotation, "Missing 'id' key in instance."
        assert "pref" in annotation, "Missing 'pref' key in instance."
        converted_instance = convert_to_dpo_format(annotation, annotation["pref"])
        if converted_instance is not None:
            converted_annotations.append(converted_instance)
    logging.info(f"Number of instances after selection: {len(converted_annotations)}")

    # Sample converted instances
    if num_instances < len(converted_annotations):
        converted_annotations = random.sample(converted_annotations, num_instances)
        logging.info(f"Sampled {num_instances} instances from the total.")

    return converted_annotations


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


if __name__ == "__main__":
    main()
