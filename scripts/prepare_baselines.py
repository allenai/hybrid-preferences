import sys
import argparse
from pathlib import Path
import logging


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
    parser.add_argument("--input_filepath", type=Path, help="Dataset path to create baselines on.")
    parser.add_argument("--num_instances", type=int, default=7000, help="Number of instances to sample.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    annotations = annotation_df.to_dict(orient="records")
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
