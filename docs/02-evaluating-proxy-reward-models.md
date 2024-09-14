# Evaluating proxy reward models

> In this guide we will **evaluate proxy reward models** on [RewardBench](https://huggingface.co/spaces/allenai/reward-bench) ([Lambert et al., 2024](https://arxiv.org/abs/2403.13787)) to get the target values for our regressor.

## Converting to Pytorch format and getting the RewardBench results

So now [you've trained a reward model, and the output gets stored in Google Cloud Storage](https://github.com/allenai/human-pref-datamodel/blob/main/docs/01-training-proxy-reward-models.md#train-reward-models-on-a-tpu).
In order to evaluate that model, we need to convert it to the Pytorch format first&mdash; easy to say but this stuff takes time and has a lot of steps (around ~15-20 minutes even on Cirrascale).

We need to run things in parallel.
To do so, we'll generate a Beaker Experiment Spec that will convert each trained reward model in GCS to Pytorch:

```sh
export DATASET=helpsteer2
python3 -m evals.generate_eval_runs \
    --template evals/template-$DATASET-counts.yml \
    --output_file $DATASET-eval-runs.yml \
    --gcs_bucket ljm-dev \
    --gcs_dir_path human-preferences/rm_checkpoints/$DATASET/tulu2_13b_rm_human_datamodel_counts
```

This will produce an `experiments.yml` file that you can use to launch several evaluation jobs at once.
For example (here's an [example run](https://beaker.org/ex/01J7Q5VGMRCHC1B3J8H7S2VWST/tasks/01J7Q5VGMYKZ85VKMG0MEWXF3J/job/01J7Q5VGT4SQSXTJX0DD3WRFYR)):

```sh
beaker experiment create helpsteer2-eval-runs.yml
```

### Appendix: What does each evaluation job do?

> You don't really need to read this, but it's just important if you want to know what's happening under the hood.
> There are some optimizations that can be done too.

Each job uses the [ljm/easylm-convert](https://beaker.org/im/01J7MR9BM7DR5EGYGMWPJ2NM47/details) image that contains tools like `gsutil`, `gcloud`, `EasyLM`, and `beaker`.
The Dockerfile for this image can be found at `evals/convert.Dockerfile`.

The important script there is the `convert_to_hf.py` file.
What it does is it:

1. Downloads a specific reward model from GCS
2. Converts the model from EasyLM to Pytorch,
3. Reuploads the model as a Beaker dataset; and
4. Launch another Beaker experiment that performs the RewardBench eval job.

Here's a single `convert_to_hf.py` job looks like:

```sh
python3 convert_to_hf.py \
    --gcs_bucket ljm-dev
    --gcs_dir_path human-preferences/rm_checkpoints/helpsteer2/tulu2_13b_rm_human_datamodel_counts_7000_ID__07add08aa33a4fa6a5294c7bc41ae1f9__SWAPS_4026--a6bf226b9e8e43f387810e0da3096526/streaming_params_437 \
    --batch_size 1 \
    --prefix helpsteer2-counts \
    --is_reward_model
```

- **Why do you need to upload the model as a Beaker dataset again if you can just store the output as a Beaker Result?** Yeah, that's how it should be done. I was running the `convert_to_hf.py` manually back then and I wasn't able to optimize the process. PRs welcome!

## Fetch all results and combine it with the features

Do you remember the `feats/*.jsonl` file you had when you generated the proxy DPO subsets? Yeah, they're back. Hope you didn't delete them!
So now we launched several jobs that evaluate on RewardBench, and we want to fetch all these results in a neat table.

To do so, we run the following command:

```sh
export DATASET=helpsteer2
python3 scripts/fetch_evals_rewardbench.py \
    --output_path $DATASET-counts-runs.csv \
    --experiment_prefix rm-eval-$DATASET-count \
    --feature_counts_dir path/to/feats/
```

The `--experiment_prefix` indicates which experiments will be fetched by script.
Most of the jobs launched from the previous conversion step has the `rm-eval-{dataset}-count` format.
To be sure, check the `--template` that you used when running `evals.generate_eval_runs`.

Finally, we'll use the CSV file as our training and development dataset for fitting the regressor.
