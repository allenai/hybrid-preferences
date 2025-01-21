import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

DPO_JOB_TEMPLATE = (
    "python3 -m EasyLM.models.llama.llama_train_dpo "
    "--mesh_dim='1,-1,4' "
    "--dtype='bf16' "
    "--num_epochs=3 "
    "--log_freq=50 "
    "--save_model_freq=1000 "
    "--save_milestone_freq=0 "
    "--load_llama_config='13b' "
    "--update_llama_config='' "
    "--load_dataset_state='' "
    "--load_checkpoint='params::{ckpt_gcs_path}' "
    "--tokenizer.vocab_file='{vocab_gcs_path}' "
    "--optimizer.type='adamw' "
    "--optimizer.adamw_optimizer.weight_decay=0.0 "
    "--optimizer.adamw_optimizer.lr=5e-7 "
    "--optimizer.adamw_optimizer.end_lr=0 "
    "--optimizer.adamw_optimizer.warmup_ratio=0.1 "
    "--optimizer.accumulate_gradient_steps=4 "
    "--train_dataset.type='preference_json_torch' "
    "--train_dataset.json_torch_dataset.path='{input_gcs_path}/{dataset_name}/{experiment_name}.jsonl' "
    "--train_dataset.json_torch_dataset.seq_length=4096 "
    "--train_dataset.json_torch_dataset.batch_size=8 "
    "--checkpointer.save_optimizer_state=False "
    "--logger.online={log_to_wandb} "
    "--logger.project='ljm-dev' "
    "--logger.entity='rlhf-llm-dev' "
    "--logger.prefix_to_id=True "
    "--logger.prefix='tulu2_13b_dpo_{experiment_name}' "
    "--logger.output_dir='{output_gcs_path}/checkpoints/{dataset_name}'"
)


RM_JOB_TEMPLATE = (
    "python3 -m EasyLM.models.llama.llama_train_rm "
    "--mesh_dim=1,-1,8 "
    "--dtype=bf16 "
    "--num_epochs=1 "
    "--log_freq=50 "
    "--save_model_freq=1000 "
    "--save_milestone_freq=0 "
    "--load_llama_config=8b31 "
    "--update_llama_config='' "
    "--load_dataset_state='' "
    "--load_checkpoint='params::{ckpt_gcs_path}' "
    "--tokenizer='meta-llama/Llama-3.1-8B' "
    "--optimizer.type=adamw "
    "--optimizer.adamw_optimizer.weight_decay=0.0 "
    "--optimizer.adamw_optimizer.lr=1e-5 "
    "--optimizer.adamw_optimizer.end_lr=1e-6 "
    "--optimizer.adamw_optimizer.warmup_ratio=0.03 "
    "--optimizer.accumulate_gradient_steps=4 "
    "--train_dataset.type=preference_json_torch "
    "--train_dataset.json_torch_dataset.path='{input_gcs_path}/{dataset_name}/{experiment_name}.jsonl' "
    "--train_dataset.json_torch_dataset.seq_length=4096 "
    "--train_dataset.json_torch_dataset.batch_size=16 "
    "--checkpointer.save_optimizer_state=False "
    "--train_dataset.json_torch_dataset.remove_truncated_samples=True "
    "--logger.online={log_to_wandb} "
    "--logger.project=ljm-dev "
    "--logger.entity=rlhf-llm-dev "
    "--logger.prefix_to_id=True "
    "--logger.prefix=llama3.1_8b_rm_{experiment_name} "
    "--logger.output_dir='{output_gcs_path}/rm_checkpoints/{dataset_name}'"
)


def get_args():
    # fmt: off
    description = """Utility CLI for easily submitting jobs to the TPU

You need to pass a TXT file, where each line is the name of the dataset to use for training.
It is recommended that the name of the dataset is the name of your experiment, so that it's easier to track.
"""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--experiment_path", type=Path, required=True, help="Path to a TXT file containing the experiments (or datasets) in a GCS bucket.")
    parser.add_argument("--tpu_name", type=str, required=True, help="Name of the TPU to run these experiments on.")
    parser.add_argument("--zone", type=str, required=True, help="Zone of the TPU.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name for managing IO paths.")
    parser.add_argument("--input_gcs_path", type=str, default="gs://ljm-dev/human-preferences/train_data", help="Path to the GCS bucket containing the datasets.")
    parser.add_argument("--output_gcs_path", type=str, default="gs://ljm-dev/human-preferences", help="Path to the GCS bucket to save the models. Will create subdirectories for DPO or RM runs.")
    parser.add_argument("--ckpt_gcs_path", type=str, default="gs://hamishi-east1/easylm/llama31/llama_3_1_8b", help="GCS filepath containing the parameter checkpoint for training.")
    parser.add_argument("--vocab_gcs_path", type=str, default="gs://hamishi-east1/easylm/llama/tokenizer.model", help="GCS filepath containing the tokenizer.")
    parser.add_argument("--train_dpo", action="store_true", default=False, help="If set, will train a DPO model instead of a Sequence Regression RM.")
    parser.add_argument("--timeout", type=int, default=300, help="Set timeout (in seconds) in between training runs.")
    parser.add_argument("--worker", type=str, default="all", help="Worker passed to the --worker argument in gcloud.")
    parser.add_argument("--log_to_wandb", action="store_true", default=False, help="If set, will log online to WandB.")
    parser.add_argument("--sort_by_swaps", action="store_true", default=False, help="If set, will prioritize running experiments with high swaps.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    experiment_path: Path = args.experiment_path
    with experiment_path.open("r") as f:
        experiment_names = f.read().splitlines()

    # Sort experiments based on the number of swaps (descending)
    if args.sort_by_swaps:
        experiment_names = sorted(
            experiment_names,
            key=lambda x: int(x.split("SWAPS_")[1].split("::")[0]),
            reverse=True,
        )

    commands_for_experiments = []
    for idx, experiment_str in enumerate(experiment_names):
        if "::" in experiment_str:
            experiment_name, _ = experiment_str.split("::")
        else:
            experiment_name = experiment_str
        if args.train_dpo:
            cmd = DPO_JOB_TEMPLATE.format(
                experiment_name=experiment_name,
                dataset_name=args.dataset_name,
                input_gcs_path=args.input_gcs_path,
                output_gcs_path=args.output_gcs_path,
                ckpt_gcs_path=args.ckpt_gcs_path,
                vocab_gcs_path=args.vocab_gcs_path,
                log_to_wandb="True" if args.log_to_wandb else "False",
            )
        else:
            cmd = RM_JOB_TEMPLATE.format(
                experiment_name=experiment_name,
                dataset_name=args.dataset_name,
                input_gcs_path=args.input_gcs_path,
                output_gcs_path=args.output_gcs_path,
                ckpt_gcs_path=args.ckpt_gcs_path,
                # vocab_gcs_path=args.vocab_gcs_path,
                log_to_wandb="True" if args.log_to_wandb else "False",
            )

        if idx < len(experiment_names) - 1:
            cmd += f" && sleep {args.timeout} && "

        commands_for_experiments.append(cmd)

    command_str = "".join(commands_for_experiments)
    logging.info(
        f"Running {len(commands_for_experiments)} commands on TPU '{args.tpu_name}':"
    )
    logging.debug(command_str)

    # Run the command using subprocess
    # tpu_command = (
    #     f"gcloud alpha compute tpus tpu-vm ssh {args.tpu_name} "
    #     "--zone=us-east1-d --project=ai2-tpu --worker=all "
    #     "--command='cd easylm; git pull; export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE';"
    #     f'echo "{command_str}" > run_experiments.sh; '
    #     "chmod +x run_experiments.sh; "
    #     "./run_experiments.sh &> experiments.log &'"
    # )

    tpu_command = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        args.tpu_name,
        f"--zone={args.zone}",
        "--project=ai2-tpu",
        f"--worker={args.worker}",
        "--command="
        + (
            "cd easylm; git pull; "
            "export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 "
            "--xla_tpu_spmd_threshold_for_allgather_cse=10000 "
            "--xla_tpu_spmd_rewrite_einsum_with_reshape=true "
            "--xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'; "
            'echo "' + command_str + '" > run_experiments.sh; '
            "chmod +x run_experiments.sh; "
            "./run_experiments.sh &> experiments.log &"
        ),
    ]

    subprocess.run(tpu_command, check=True)
    logging.info(
        "TPU command sent. You can track the logs by using the following command: \n"
        f'gcloud alpha compute tpus tpu-vm ssh {args.tpu_name} --worker={args.worker} --zone={args.zone} --project=ai2-tpu --command="tail -f easylm/experiments.log"'
    )


if __name__ == "__main__":
    main()
