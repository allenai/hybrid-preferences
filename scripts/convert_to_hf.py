"""A semi-portable script for converting many files from EasyLM to HF format

Usage:

```
# Setup EasyLM
git clone https://github.com/hamishivi/EasyLM.git
cd EasyLM
gsutil cp gs://hamishi-east1/easylm/llama/tokenizer.model .
conda env create -f scripts/gpu_environment.yml
conda activate EasyLM
pip install google-cloud-storage beaker-py
gcloud auth application-default login
gcloud auth application login
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
    parser.add_argument("--download_dir", type=Path, help="Directory where all parameter downloads from GCS will be stored. Will be emptied for every batch.")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.model", help="Path where you downloaded the tokenizer model.")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of models to download before deleting.")
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
        download_dir = Path(args.download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"*** Processing batch: {idx+1} ***")
        with open("src_files.txt", "w") as f:
            for line in batch:
                filedir = Path(line).parent
                f.write(f"gs://{args.gcs_bucket}/{filedir}\n")

        download_command = f"cat src_files.txt | gsutil -m cp -I -r {download_dir}"
        subprocess.run(download_command, text=True, shell=True, capture_output=False)

    # Create output file and save outputs there
    # param_file = "streaming_params"
    # download_path = param_dir / param_file
    # gcs_path.chunk_size = 4 * 1024 * 1024
    # gcs_path.download_to_filename(str(download_path))
    # params_dict[str(download_path)] = out_dir

    # for idx, (input_params, output_dir) in enumerate(params_dict.items()):
    #     logging.info(f"Converting {download_path} to HF format")
    #     convert_command = [
    #         "python",
    #         "-m",
    #         "EasyLM.models.llama.convert_easylm_to_hf",
    #         f"--load_checkpoint=params::{input_params}",
    #         f"--tokenizer_path={args.tokenizer_path}",
    #         f"--model_size={args.model_size}",
    #         f"--output_dir={output_dir}",
    #     ]
    #     subprocess.run(convert_command, check=True)


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


if __name__ == "__main__":
    main()
