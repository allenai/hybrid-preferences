"""This script is a copy of the run_rm.py pipeline from RewardBench

https://github.com/allenai/reward-bench/blob/main/scripts/run_rm.py

We want to use the same model config and set-up, but change the evaluation dataset.
The RewardBench codebase tightly integrates the benchmark with the code, and it's not straightforward to change the target without forking the code.
"""

import argparse
import logging
import json
import os
import sys
import hashlib
from pathlib import Path
from typing import Any

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset, Dataset
from fastchat.conversation import get_conv_template, Conversation
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.rm_inference import RewardBenchPipeline

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

# Enable TensorFloat32 (TF32) tensor cores on Ampere GPUs for matrix multiplications (faster than FP32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Get evaluation scores for Helpsteer2 Dev Set")
    parser.add_argument("--model", type=Path, help="Path to the model to evaluate.")
    parser.add_argument("--tokenizer", type=Path, help="Path to the tokenizer.")
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--chat_template", type=str, default="tulu", help="Chat template to use to format the preference instance.")
    parser.add_argument("--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline.")
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging).")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length of RM inputs (passed to pipeline)")
    parser.add_argument("--disable_beaker_save", action="store_true", help="disable saving the main results in a file for AI2 Beaker.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    ###############
    # Setup logging
    ###############
    accelerator = Accelerator()
    current_device = accelerator.process_index

    logger = get_logger(__name__)
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Running RM on {args.model} with chat template {args.chat_template}")
    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    chat_template = args.chat_template
    conv = get_conv_template(chat_template)

    # Just use default
    config = {
        "model_builder": AutoModelForSequenceClassification.from_pretrained,
        "pipeline_builder": RewardBenchPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    }

    quantized = config["quantized"]

    custom_dialogue = config["custom_dialogue"]
    model_type = config["model_type"]
    model_builder = config["model_builder"]
    pipeline_builder = config["pipeline_builder"]
    torch_dtype = config.get("torch_dtype", None)
    # if not datatype in config (default), check args
    if torch_dtype is None:
        # if datatype is bfloat16, then manually turn off quantizaiton (done with bitsandbytes)
        if args.torch_dtype == torch.bfloat16:
            quantized = False
            logger.info("Disabling quantization for bfloat16 datatype")
        torch_dtype = args.torch_dtype

    # not included in config to make user explicitly understand they are passing this
    trust_remote_code = args.trust_remote_code

    ############################
    # Load dataset
    ############################
    logging.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=args.trust_remote_code
    )
    if not custom_dialogue:  # not needed for PairRM / SteamSHP
        # copied from Starling, but few samples are above context length
        tokenizer.truncation_side = "left"

    dataset = load_helpsteer2_dataset(
        dataset_path="nvidia/Helpsteer2",
        split="validation",
        weights="llama",
        conv=conv,
        keep_columns=["text_chosen", "text_rejected", "id"],
    )

    # copy id for saving, then remove
    ids = dataset["id"]
    dataset = dataset.remove_columns("id")

    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size
    logger.info("*** Load reward model ***")
    reward_pipeline_kwargs = {
        "batch_size": BATCH_SIZE,  # eval_args.inference_batch_size,
        "truncation": True,
        "padding": True,
        "max_length": args.max_length,
        "function_to_apply": "none",  # Compute raw logits
        "return_token_type_ids": False,
    }
    if quantized:
        model_kwargs = {
            "load_in_8bit": True,
            "device_map": {"": current_device},
            "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
        }
    else:
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch_dtype,
        }

    # if attn_implementation is not specified, this falls back to Hugging Face's default
    # strategy (which chooses between sdpa and eager depending on pytorch version)
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = model_builder(
        args.model, **model_kwargs, trust_remote_code=trust_remote_code
    )
    reward_pipe = pipeline_builder(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
    )

    ############################
    # Tokenization settings & dataset preparation
    ############################
    # set pad token to eos token if not set
    if reward_pipe.tokenizer.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
        reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id
    # For models whose config did not contains `pad_token_id`
    if reward_pipe.model.config.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.pad_token_id

    # if using fastchat template (no template in tokenizer), make the RM tokenizer output an EOS token
    if not check_tokenizer_chat_template(tokenizer):
        reward_pipe.tokenizer.add_eos_token = True

    logger.info("*** Running forward pass via built in pipeline abstraction ***")
    # this setup can be optimized slightly with one pipeline call
    # prepare for inference
    reward_pipe = accelerator.prepare(reward_pipe)

    results_rej = reward_pipe(dataset["text_rejected"], **reward_pipeline_kwargs)
    results_cho = reward_pipe(dataset["text_chosen"], **reward_pipeline_kwargs)

    # extract scores from results which is list of dicts, e.g. [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
    scores_chosen = [result["score"] for result in results_cho]
    scores_rejected = [result["score"] for result in results_rej]

    # pairwise comparison list comprehension
    results = [
        1 if chosen > rejected else 0
        for chosen, rejected in zip(scores_chosen, scores_rejected)
    ]

    ############################
    # Print & process results
    ############################
    # add column for results for easy printing
    out_dataset = dataset.add_column("results", results)
    out_dataset = out_dataset.add_column("id", ids)
    out_dataset = out_dataset.add_column("scores_chosen", scores_chosen)
    out_dataset = out_dataset.add_column("scores_rejected", scores_rejected)

    with open("metrics.json", "w") as f:
        scores = {"hs2_dev_accuracy": sum(results) / len(results)}
        json.dump(scores, f)


def check_tokenizer_chat_template(tokenizer):
    """
    Check if tokenizer has non none chat_template attribute.
    """
    if hasattr(tokenizer, "chat_template"):
        if tokenizer.chat_template is not None:
            return True
    return False


def load_helpsteer2_dataset(
    conv: Conversation,
    dataset_path: str = "nvidia/Helpsteer2",
    split: str = "validation",
    weights: str = "llama",
    keep_columns: list[str] = ["text_chosen", "text_rejected", "id"],
):
    if weights == "llama":
        wt_vec = {
            "helpfulness": 0.65,
            "correctness": 0.8,
            "coherence": 0.45,
            "complexity": 0.55,
            "verbosity": -0.4,
        }
    elif weights == "nemotron":
        wt_vec = {
            "helpfulness": 0.3,
            "correctness": 0.74,
            "coherence": 0.46,
            "complexity": 0.47,
            "verbosity": -0.33,
        }
    else:
        raise ValueError("Unknown weights. Please pass either 'llama' or 'nemotron'")

    # Binarize the dataset
    init_dataset = load_dataset(dataset_path, split=split)
    df = init_dataset.to_pandas()

    def _compute_rating(row, wt_vec):
        return sum(row[col] * wt_vec[col] for col in wt_vec)

    df["rating"] = df.apply(_compute_rating, wt_vec=wt_vec, axis=1)
    df["response_group"] = df.groupby("prompt").cumcount()
    df_binary = df.pivot(
        index="prompt", columns="response_group", values=["response", "rating"]
    )
    df_binary.columns = ["response_a", "response_b", "rating_a", "rating_b"]
    df_binary["chosen"] = df_binary.apply(
        lambda row: (
            row["response_a"]
            if row["rating_a"] > row["rating_b"]
            else row["response_b"]
        ),
        axis=1,
    )
    df_binary["rejected"] = df_binary.apply(
        lambda row: (
            row["response_b"]
            if row["rating_a"] > row["rating_b"]
            else row["response_a"]
        ),
        axis=1,
    )

    df_binary = df_binary.reset_index()
    df_binary["id"] = df_binary["prompt"].apply(
        lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()
    )

    raw_dataset = Dataset.from_pandas(df_binary)
    logging.info("*** Preparing dataset with FastChat ***")
    dataset = raw_dataset.map(
        prepare_dialogue,
        fn_kwargs={"dialogue_template": conv},
        num_proc=8,
        load_from_cache_file=False,
    )

    all_cols = dataset.column_names
    dataset = dataset.remove_columns([c for c in all_cols if c not in keep_columns])
    return dataset


def prepare_dialogue(
    example: dict[str, Any],
    dialogue_template: Conversation,
    ift: bool = False,
) -> dict[str, Any]:
    """Format example to single- or multi-turn dialogue."""
    if all(k in example.keys() for k in ("chosen", "rejected")):
        # multi turn
        if isinstance(example["prompt"], list) and len(example["prompt"]) > 0:
            # iterate through prompt messages, alternate user and assistant, end with example["chosen"]/rejected
            dialogue_template.messages = []
            for i, (line) in enumerate(example["prompt"]):
                p = line["content"]
                _ = line["role"]
                if (i + 1) % 2 == 1:
                    dialogue_template.messages.append([dialogue_template.roles[0], p])
                else:
                    dialogue_template.messages.append([dialogue_template.roles[1], p])
            # assert that the last message before this is user
            assert dialogue_template.messages[-1][0] == dialogue_template.roles[0]

            # needed for DPO
            temp_prompt = dialogue_template.get_prompt()

            # end with chosen/rejected
            dialogue_template.messages.append(
                [dialogue_template.roles[1], example["chosen"]]
            )
            example["text_chosen"] = dialogue_template.get_prompt()

            dialogue_template.messages[-1] = [
                dialogue_template.roles[1],
                example["rejected"],
            ]
            example["text_rejected"] = dialogue_template.get_prompt()

            example["prompt"] = temp_prompt

        # single turn
        else:
            if isinstance(example["prompt"], list):
                example["prompt"] = example["prompt"][0]
            dialogue_template.messages = [
                [dialogue_template.roles[0], example["prompt"]],
            ]
            temp_prompt = dialogue_template.get_prompt()

            dialogue_template.messages = [
                [dialogue_template.roles[0], example["prompt"]],
                [dialogue_template.roles[1], example["chosen"]],
            ]
            example["text_chosen"] = dialogue_template.get_prompt()
            dialogue_template.messages = [
                [dialogue_template.roles[0], example["prompt"]],
                [dialogue_template.roles[1], example["rejected"]],
            ]
            example["text_rejected"] = dialogue_template.get_prompt()

            example["prompt"] = temp_prompt
    elif ift:
        if isinstance(example["prompt"], list):
            example["prompt"] = example["prompt"][0]

        dialogue_template.messages = [
            [dialogue_template.roles[0], example["prompt"]],
        ]
        temp_prompt = dialogue_template.get_prompt()
        dialogue_template.messages = [
            [dialogue_template.roles[0], example["prompt"]],
            [dialogue_template.roles[1], example["input"]],
        ]
        example["text"] = dialogue_template.get_prompt()
        example["prompt"] = temp_prompt  # needed for DPO

    else:
        raise ValueError(
            "Could not format example as dialogue for `rm` task!"
            f"Require `[chosen, rejected]` keys but found {list(example.keys())}"
        )
    return example


if __name__ == "__main__":
    # main()
    chat_template = "tulu"
    conv = get_conv_template(chat_template)
    dataset = load_helpsteer2_dataset(
        dataset_path="nvidia/Helpsteer2",
        split="validation",
        weights="llama",
        conv=conv,
        keep_columns=["text_chosen", "text_rejected", "id"],
    )
