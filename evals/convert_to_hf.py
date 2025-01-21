"""A semi-portable script for converting many files from EasyLM to HF format

Usage:

```
# Setup EasyLM
git clone https://github.com/hamishivi/EasyLM.git
cd EasyLM
git checkout bc241782b67bbe926e148ec9d2046d76b7ba58c8 .
conda env create -f scripts/gpu_environment.yml
gcloud auth login
gsutil cp gs://hamishi-east1/easylm/llama/tokenizer.model .
conda run -n EasyLM pip install google-cloud-storage beaker-py
conda run -n EasyLM pip install huggingface-hub --upgrade
gcloud auth application-default login
gcloud set project ai2-allennlp
beaker account login
# Copy this script into the machine you're working on
conda run -n EasyLM python convert_to_hf.py --gcs_bucket <BUCKET_NAME> --gcs_dir_path <PREFIX> --parent_dir <OUTPUT>
```

"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from beaker.client import Beaker, Constraints, DataMount, DataSource, EnvVar
from beaker.client import ExperimentSpec, ImageSource, ResultSpec, TaskContext
from beaker.client import TaskResources, TaskSpec

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
    parser.add_argument("--gcs_bucket", type=str, help="GCS bucket where the models are stored (NO need for gs:// prefix).")
    parser.add_argument("--gcs_dir_path", type=str, help="The directory path (or prefix) of models (e.g., human-preferences/rm_checkpoints/tulu2_13b_rm_human_datamodel_).")
    parser.add_argument("--download_dir", type=Path, default="download_dir", help="Parent directory where all parameter downloads from GCS will be stored. Ephemerable: will be emptied for every batch.")
    parser.add_argument("--prefix", type=str, default=None, help="Custom prefix to further differentiate experiments.")
    parser.add_argument("--pytorch_dir", type=Path, default="pytorch_dir", help="Parent directory to store all converted pytorch files. Ephemerable: will be emptied for every batch.")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.model", help="Path where you downloaded the tokenizer model.")
    parser.add_argument("--model_size", type=str, default="13b", help="Model size to pass to EasyLM.")
    parser.add_argument("--batch_size", type=int, default=3, help="Number of models to download before deleting.")
    parser.add_argument("--is_reward_model", default=False, action="store_true", help="Set if converting a reward model.")
    parser.add_argument("--beaker_workspace", default="ai2/ljm-oe-adapt", help="Beaker workspace to upload datasets.")
    parser.add_argument("--max_workers", default=1, type=int, help="Number of workers to spawn when uploading to Beaker datasets.")
    parser.add_argument("--force", action="store_true", default=False, help="If force, just convert the model.")
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
    logging.info(f"Converting into batches of {args.batch_size} to save space")

    # Do not process files that were already done by
    # checking if the dataset in Beaker already exists
    existing_datasets = [d.name for d in beaker.workspace.datasets(match="tulu2_13b")]
    if args.force:
        exp_names = [Path(s).parents[0].name.split("--")[0] for s in src_files]
        diff = [
            src_file
            for src_file, experiment in zip(src_files, exp_names)
            if not any(experiment in b_item for b_item in existing_datasets)
        ]
        logging.info(f"Found {len(src_files) - len(diff)} datasets already done!")
        src_files = diff
    logging.info(f"Running {len(src_files)} experiments.")
    batches = make_batch(src_files, batch_size=args.batch_size)

    for idx, batch in enumerate(batches):

        # Perform download in batches to save disk space
        download_dir = Path(args.download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"*** Processing batch: {idx+1} ***")
        with open("src_files.txt", "w") as f:
            for line in batch:
                filedir = Path(line).parent
                f.write(f"gs://{args.gcs_bucket}/{filedir}\n")

        download_command = f"cat src_files.txt | gsutil -m cp -I -r {download_dir}"
        logging.info("Downloading files")
        logging.info(f"Running command: {download_command}")
        subprocess.run(download_command, text=True, shell=True, capture_output=False)

        # Convert output from GCS to HuggingFace format
        logging.info("Converting to HF format")
        params_paths: list[Path] = find_dirs_with_files(
            download_dir, "*streaming_params*"
        )
        pytorch_dir = Path(args.pytorch_dir)
        for params_path in params_paths:
            if "llama" in str(params_path):
                experiment_name = params_path.parts[-2].replace(".", "-").split("--")[0]
            else:
                experiment_name = params_path.parent.stem.split("--")[0]
            if args.prefix:
                experiment_name = f"{args.prefix}-{experiment_name}"
            output_dir = pytorch_dir / experiment_name
            output_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Saving to {output_dir}")
            convert_command = [
                "python",
                "-m",
                "EasyLM.models.llama.convert_easylm_to_hf",
                f"--load_checkpoint=params::{params_path}",
                f"--tokenizer_path={args.tokenizer_path}",
                f"--model_size={args.model_size}",
                f"--output_dir={output_dir}",
            ]

            if args.is_reward_model:
                logging.info("Passing --is_reward_model flag")
                convert_command += ["--is_reward_model"]

            logging.info(f"Running command: {convert_command}")
            subprocess.run(convert_command, check=True)

            # Upload each converted model to beaker so we can run evaluations there
            logging.info("Pushing to beaker...")
            description = f"Human data model for experiment: {experiment_name}"
            description += " (RM)" if args.is_reward_model else " (DPO)"

            # Try using `beaker dataset create` because the python API doesn't work
            upload_command = [
                "beaker",
                "dataset",
                "create",
                output_dir,
                "--desc",
                description,
                "--name",
                experiment_name,
                "--workspace",
                args.beaker_workspace,
            ]
            logging.info(f"Running command: {upload_command}")
            try:
                subprocess.run(upload_command, check=True)
            except Exception as e:
                logging.error(f"Error found: {e}")
                break

            # dataset = beaker.dataset.create(
            #     experiment_name,
            #     output_dir,
            #     description=description,
            #     force=True,
            #     strip_paths=True,
            #     max_workers=args.max_workers,
            # )
            # logging.info(f"Uploaded dataset to {dataset.id}")

            # Create experiment and auto-queue
            logging.info("Sending eval script to beaker...")
            spec = create_beaker_experiment_spec(
                experiment_name=experiment_name,
                reward_model_beaker_id=experiment_name,
                account_name=account.name,
            )
            experiment = beaker.experiment.create(
                spec=spec,
                name=f"rm-eval-{experiment_name}",
                workspace=args.beaker_workspace,
            )
            logging.info(f"Running experiment {experiment.id}")

        # Delete contents where we downloaded the model and where converted them
        # in order to save space. Then we go ahead with the next batch
        logging.info("Emptying directories as preparation for next batch...")
        rmtree(download_dir)
        rmtree(pytorch_dir)


def make_batch(l: list[Any], batch_size: int) -> list[list[Any]]:
    return [l[i : i + batch_size] for i in range(0, len(l), batch_size)]


def list_directories_with_prefix(
    bucket_name: list[str], prefix: list[str]
) -> list["storage.Blob"]:
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    # Use the prefix to filter objects
    blobs = bucket.list_blobs(prefix=prefix)

    # Extract directories
    directories = list()
    directories = [blob for blob in blobs if "streaming_params" in blob.name]
    return directories


def find_dirs_with_files(base_dir: Path, pattern: str):
    matching_dirs = set()

    # Iterate over all files matching the pattern
    for file in base_dir.rglob(pattern):
        matching_dirs.add(file)

    return list(matching_dirs)


def create_beaker_experiment_spec(
    experiment_name: str,
    reward_model_beaker_id: str,
    account_name: Optional[str] = None,
) -> ExperimentSpec:
    dataset_name = (
        f"{account_name}/{reward_model_beaker_id}"
        if account_name
        else reward_model_beaker_id
    )
    spec = ExperimentSpec(
        budget="ai2/oe-adapt",
        version="v2",
        description="Perform rewardbench evaluation",
        tasks=[
            TaskSpec(
                name=f"evaluate-{experiment_name}",
                image=ImageSource(beaker="nathanl/rewardbench_auto"),
                constraints=Constraints(
                    cluster=[
                        "ai2/saturn-cirrascale",
                        "ai2/ceres-cirrascale",
                        "ai2/jupiter-cirrascale-2",
                    ]
                ),
                context=TaskContext(priority="normal", preemptible=True),
                result=ResultSpec(path="/output"),
                command=["/bin/sh", "-c"],
                arguments=[
                    "python scripts/run_rm.py --model /reward_model --tokenizer /reward_model --batch_size 8 --trust_remote_code --do_not_save"
                ],
                datasets=[
                    DataMount(
                        source=DataSource(beaker=dataset_name),
                        mount_path="/reward_model",
                    ),
                    # There's no more NFS but we'll keep this here for posterity
                    # DataMount(
                    #     source=DataSource(host_path="/net/nfs.cirrascale"),
                    #     mount_path="/net/nfs.cirrascale",
                    # ),
                ],
                resources=TaskResources(gpu_count=1),
                env_vars=[
                    EnvVar(name="CUDA_DEVICE_ORDER", value="PCI_BUS_ID"),
                    EnvVar(name="TRANSFORMERS_CACHE", value="./cache/"),
                    EnvVar(name="WANDB_WATCH", value="false"),
                    EnvVar(name="WANDB_LOG_MODEL", value="false"),
                    EnvVar(name="WANDB_DISABLED", value="true"),
                    EnvVar(name="WANDB_PROJECT", value="rewardbench"),
                    EnvVar(name="HF_TOKEN", secret="HF_TOKEN"),
                ],
            )
        ],
    )
    return spec


def rmtree(f: Path):
    if f.is_file():
        f.unlink()
    else:
        for child in f.iterdir():
            rmtree(child)
        f.rmdir()


if __name__ == "__main__":
    main()
