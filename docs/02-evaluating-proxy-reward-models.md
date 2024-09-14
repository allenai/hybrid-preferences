## Evaluating proxy reward models

> In this guide we will **evaluate proxy reward models** on [RewardBench](https://huggingface.co/spaces/allenai/reward-bench) ([Lambert et al., 2024](https://arxiv.org/abs/2403.13787)) to get the target values for our regressor.

So if you've trained a reward model, the output gets stored in Google Cloud Storage.
In order to evaluate that model, we need to convert it to the Pytorch format first.

Downloading GCS files, converting, and re-uploading to Beaker take some time (around ~15-20 minutes each on a good CPU).
We need to run things in parallel.
To do so, we'll generate a Beaker Experiment Spec that will convert each trained reward model in GCS:

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

### What does each evaluation job do?

> You don't really need to read this, but it's just important if you want to know what's happening in the job.

Each job uses the [ljm/easylm-convert](https://beaker.org/im/01J7MR9BM7DR5EGYGMWPJ2NM47/details) image that contains tools like `gsutil`, `gcloud`, `EasyLM`, and `beaker`.
The Dockerfile for this image can be found at `evals/convert.Dockerfile`.

The important script there is the `convert_to_hf.py` file.
What it does is it (1) downloads a specific reward model from GCS, (2) converts the model from EasyLM to Pytorch, (3) reuploads the model as a Beaker dataset, and (4) start a RewardBench eval job using
