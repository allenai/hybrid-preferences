import argparse
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm

from beaker.client import Beaker
from beaker.client import ExperimentSpec

from evals.convert_to_hf import list_directories_with_prefix

try:
    from google.cloud import storage
except ModuleNotFoundError:
    print("Install GCS Python client via:\n\n`pip install google-cloud-storage`\n")
    print("Then, authenticate via:\n\n`gcloud auth application-default login`")
    raise

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", help="Experiment YAML template.")
    parser.add_argument("--output_file", help="Path to save the experiment YAML file to run.")
    parser.add_argument("--gcs_bucket", type=str, help="GCS bucket where the models are stored (NO need for gs:// prefix).")
    parser.add_argument("--gcs_dir_path", type=str, help="The directory path (or prefix) of models (e.g., human-preferences/rm_checkpoints/tulu2_13b_rm_human_datamodel_).")
    parser.add_argument("--prefix", type=str, help="Prefix to append to the eval runs.")
    parser.add_argument("--is_reward_model", action="store_true", default="If set, will train a reward model.")
    parser.add_argument("--beaker_workspace", default="ai2/ljm-oe-adapt", help="Beaker workspace to upload datasets.")
    parser.add_argument("--cleanup", action="store_true", default=False, help="If set, will delete uncommitted datasets (make sure no other jobs are running!)")
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

    params_gcs_paths: list["storage.Blob"] = list_directories_with_prefix(
        bucket_name=args.gcs_bucket, prefix=args.gcs_dir_path
    )
    logging.info(f"Found {len(params_gcs_paths)} parameter files.")

    src_files = [gcs_path.name for gcs_path in params_gcs_paths]

    # Do not process files that were already done by
    # checking if the dataset in Beaker already exists
    existing_datasets = [
        d.name for d in beaker.workspace.datasets(match="tulu2_13b", uncommitted=False)
    ]

    # Delete datasets that weren't committed
    if args.cleanup:
        logging.info("Deleting uncommitted datasets")
        uncommited_datasets = beaker.workspace.datasets(
            match="tulu2_13b", uncommitted=True
        )
        for uncommited_dataset in tqdm(
            uncommited_datasets, total=len(uncommited_datasets)
        ):
            logging.debug(f"Deleting {uncommited_dataset.name}")
            beaker.dataset.delete(uncommited_dataset)
            time.sleep(3)

    exp_names = [Path(s).parents[0].name.split("--")[0] for s in src_files]
    diff = [
        src_file
        for src_file, experiment in zip(src_files, exp_names)
        if not any(experiment in b_item for b_item in existing_datasets)
    ]
    logging.info(f"Found {len(src_files) - len(diff)} datasets already done!")
    src_files = diff
    logging.info(f"Generating experiment file for {len(src_files)} experiments.")

    spec = ExperimentSpec.from_file(args.template)
    exp_spec = deepcopy(spec)
    template_task = exp_spec.tasks.pop(0)

    new_tasks = []
    for idx, src_file in enumerate(src_files):
        task = deepcopy(template_task)
        task.name = f"convert-and-run-evals-{idx}"
        task.arguments.extend(["--gcs_dir_path"] + [src_file])
        task.arguments.extend(["--prefix"] + [args.prefix])
        if args.is_reward_model:
            task.arguments.extend(["--is_reward_model"])
        new_tasks.append(task)

    exp_spec.tasks = new_tasks
    exp_spec.validate()
    exp_spec.to_file(args.output_file)


if __name__ == "__main__":
    main()
