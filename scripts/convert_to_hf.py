"""A semi-portable script for converting many files from EasyLM to HF format

Usage:

```
# Setup EasyLM
git clone https://github.com/hamishivi/EasyLM.git
cd EasyLM
git checkout bc241782b67bbe926e148ec9d2046d76b7ba58c8 .
conda env create -f scripts/gpu_environment.yml
conda activate EasyLM
gcloud auth login
gsutil cp gs://hamishi-east1/easylm/llama/tokenizer.model .
pip install google-cloud-storage beaker-py
pip install huggingface-hub --upgrade
gcloud auth application-default login
# Copy this script into the machine you're working on
python convert_to_hf.py --gcs_bucket <BUCKET_NAME> --gcs_dir_path <PREFIX> --parent_dir <OUTPUT>
```

"""

import sys
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Any

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
    parser.add_argument("--pytorch_dir", type=Path, default="pytorch_dir", help="Parent directory to store all converted pytorch files. Ephemerable: will be emptied for every batch.")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.model", help="Path where you downloaded the tokenizer model.")
    parser.add_argument("--model_size", type=str, default="13b", help="Model size to pass to EasyLM.")
    parser.add_argument("--batch_size", type=int, default=3, help="Number of models to download before deleting.")
    parser.add_argument("--is_reward_model", default=False, action="store_true", help="Set if converting a reward model.")
    parser.add_argument("--beaker_workspace", default=None, help="Beaker workspace to upload datasets.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    params_gcs_paths: list["storage.Blob"] = list_directories_with_prefix(
        bucket_name=args.gcs_bucket, prefix=args.gcs_dir_path
    )
    logging.info(f"Found {len(params_gcs_paths)} parameter files.")

    src_files = [gcs_path.name for gcs_path in params_gcs_paths]
    batches = make_batch(src_files, batch_size=args.batch_size)
    logging.info(f"Converting into batches of {args.batch_size} to save space")

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
            experiment_name = params_path.parent.stem.split("--")[0]
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
            breakpoint()

            # Upload each converted model to beaker so we can run evaluations there
            if args.beaker_workspace:
                logging.info("Pushing to beaker")
                beaker = Beaker.from_env(default_workspace=args.beaker_workspace)
                description = f"Human data model for experiment: {experiment_name}"
                description += " (RM)" if args.is_reward_model else " (DPO)"
                dataset = beaker.dataset.create(
                    experiment_name,
                    output_dir,
                    description=description,
                    force=True,
                )

            logging.info("Sending eval script to beaker")
            # TODO:


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


if __name__ == "__main__":
    main()
