"""A semi-portable script for converting files from EasyLM to HF format

Usage:

```
pip install google-cloud-storage beaker-py
gcloud auth application-default login
git clone https://github.com/hamishivi/EasyLM.git
cd EasyLM
# Copy this script into the machine you're working on
python convert_to_hf.py --gcs_bucket <BUCKET_NAME> --gcs_dir_path <PREFIX> --parent_dir <OUTPUT>
```

"""

import sys
import argparse
import logging
from pathlib import Path

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
    parser.add_argument("--parent_dir", type=Path, help="Directory where all downloads and outputs will be stored.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    parent_dir = Path(args.parent_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)

    params_gcs_paths: list["storage.Blob"] = list_directories_with_prefix(
        bucket_name=args.gcs_bucket, prefix=args.gcs_dir_path
    )
    logging.info(f"Found {len(params_gcs_paths)} parameter files.")

    params_dict = {}
    for idx, gcs_path in enumerate(params_gcs_paths):
        # Create experiment name as an ID
        experiment_name = Path(gcs_path.name).parent.stem.split("--")[0]
        logging.info(
            f"*** Downloading {experiment_name} ({idx+1} / {len(params_gcs_paths)}) ***"
        )
        # Input and output directories
        param_dir = parent_dir / experiment_name
        out_dir = parent_dir / f"{experiment_name}_OUT"
        param_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Create output file and save outputs there
        param_file = "streaming_params"
        download_path = param_dir / param_file
        gcs_path.chunk_size = 4 * 1024 * 1024
        gcs_path.download_to_filename(str(download_path))
        params_dict[download_path] = out_dir

    for idx, (input_params, output_dir) in enumerate(params_dict.items()):
        pass


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
