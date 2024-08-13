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

DPO_JOB_TEMPLATE = """"""

RM_JOB_TEMPLATE = """
python3 -m EasyLM.models.llama.llama_train_rm \
        --mesh_dim=1,-1,8 \
        --dtype=bf16 \
        --num_epochs=1 \
        --log_freq=50 \
        --save_model_freq=1000 \
        --save_milestone_freq=0 \
        --load_llama_config=13b \
        --update_llama_config='' \
        --load_dataset_state='' \
        --load_checkpoint=params::{ckpt_gcs_path} \
        --tokenizer.vocab_file={vocab_gcs_path} \
        --optimizer.type=adamw \
        --optimizer.adamw_optimizer.weight_decay=0.0 \
        --optimizer.adamw_optimizer.lr=1e-5 \
        --optimizer.adamw_optimizer.end_lr=1e-6 \
        --optimizer.adamw_optimizer.warmup_ratio=0.03 \
        --optimizer.accumulate_gradient_steps=4 \
        --train_dataset.type=preference_json_torch \
        --train_dataset.json_torch_dataset.path='{input_gcs_path}{experiment_name}.jsonl' \
        --train_dataset.json_torch_dataset.seq_length=4096 \
        --train_dataset.json_torch_dataset.batch_size=16 \
        --checkpointer.save_optimizer_state=False \
        --train_dataset.json_torch_dataset.remove_truncated_samples=True \
        --logger.online=True \
        --logger.project=human_preferences_rm \
        --logger.entity=rlhf-llm-dev \
        --logger.prefix_to_id=True \
        --logger.prefix=tulu2_13b_rm_${experiment_name} \
        --logger.output_dir='{output_gcs_path}/rm_checkpoints/'
"""


def get_args():
    # fmt: off
    description = """Utility CLI for easily submitting jobs to the TPU

You need to pass a TXT file, where each line is the name of the dataset to use for training.
It is recommended that the name of the dataset is the name of your experiment, so that it's easier to track.
"""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--experiment_path", type=Path, required=True, help="Path to a TXT file containing the experiments (or datasets) in a GCS bucket.")
    parser.add_argument("--tpu_name", type=str, required=True, help="Name of the TPU to run these experiments on.")
    parser.add_argument("--input_gcs_path", type=str, default="gs://ljm-dev/human-preferences/train_data/", help="Path to the GCS bucket containing the datasets.")
    parser.add_argument("--output_gcs_path", type=str, default="gs://ljm-dev/human-preferences/", help="Path to the GCS bucket to save the models. Will create subdirectories for DPO or RM runs.")
    parser.add_argument("--ckpt_gcs_path", type=str, default="gs://hamishi-east1/easylm/llama2/tulu2_13b_fixed/tulu2_13b_fixed/455af914503740be9664497dae996762/streaming_params", help="GCS filepath containing the parameter checkpoint for training.")
    parser.add_argument("--vocab_gcs_path", type=str, default="gs://hamishi-east1/easylm/llama/tokenizer.model", help="GCS filepath containing the tokenizer.")
    parser.add_argument("--train_dpo", action="store_true", default=False, help="If set, will train a DPO model instead of a Sequence Regression RM.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    experiment_path: Path = args.experiment_path
    with experiment_path.open("r") as f:
        experiment_names = f.read().splitlines()

    commands_for_experiments = []
    for idx, experiment_name in enumerate(experiment_names):
        if args.is_dpo:
            cmd = DPO_JOB_TEMPLATE.format()
        else:
            cmd = RM_JOB_TEMPLATE.format(
                experiment_name=experiment_name,
                input_gcs_path=args.input_gcs_path,
                output_gcs_path=args.output_gcs_path,
                ckpt_gcs_path=args.ckpt_gcs_path,
                vocab_gcs_path=args.vocab_gcs_path,
            )

        if idx < len(experiment_names) - 1:
            cmd += " && sleep 900 && "

        commands_for_experiments.append(cmd)

    command_str = "".join(commands_for_experiments)
    logging.info(f"Running {len(commands_for_experiments)} commands on TPU:")
    logging.debug(command_str)


if __name__ == "__main__":
    main()
