# Training proxy reward models

> In this guide, we will train **proxy reward models** on a Google Cloud Engine TPU.

These instructions are also the same when you start training the best mixes, or any other reward / DPO models you need in your experiment.

## Create proxy DPO training datasets

First, we need to create **proxy DPO training datasets** that simulate different mixes of direct and synthetic human preferences.

```sh
python -m scripts.get_count_feats \
    --input_path path/to/features.jsonl \
    --output_dir path/to/output/dir \
    --n_train_instances 500 \
    --n_samples 7000
```

In the command above, `--n_train_instances` indicate the number of proxy DPO training datasets to create.
It will sample a budget from 1 to the size of the dataset, and then based on a feature count and a knapsack algorithm, select between GPT-4 or human annotations.
The alternative is to run `beaker/get_count_features.yml`.

This command will produce three types of artifacts in the output directory:

- The `experiments.txt` file that we will use for sending TPU jobs.
- The `feats/*.jsonl` files that we will use later on to fit a regressor. Each file is a training data point X. We don't have the target values yet (i.e., the actual reward model performance), we will get it later. **These files are important, remember that we'll need them later on!**
- The `counts/*.jsonl` files that we need to upload in Google Cloud Storage. This will be read by the EasyLM package and we'll use it to train several reward models.

## Train reward models on a TPU

### Submitting TPU jobs

You need to upload the JSONL datasets in Google Cloud Storage.
In addition, you also need to get the `experiments.txt` file as this automatically lists all experiments we want to run in the TPU.
Remember, the name of the dataset will also be the name of the experiment.

First, create and setup the TPU environment:

```sh
export TPU_NAME=ljm-v3-128-1
export TPU_TYPE=v3-128
export TPU_ZONE=us-east1-d
export GCP_PROJECT=ai2-tpu
WANDB_TOKEN=<your wandb token> scripts/create_tpu_single.sh $TPU_NAME $TPU_TYPE $TPU_ZONE
```

Once this is done, you can start submitting jobs. Below is an example run:

```sh
python -m scripts.submit_tpu_train_job \
    --experiment_path path/to/experiments.txt \
    --tpu_name $TPU_NAME \
    --zone $TPU_ZONE \
    --log_to_wandb
    # Pass this if you want to train DPO model
    # --train_dpo
```

To see the progress or training logs, you can either check wandb or run the following command:

```sh
gcloud alpha compute tpus tpu-vvm ssh $TPU_NAME \
    --worker=all \
    --zone=$TPU_ZONE \
    --project=$GCP_PROJECT \
    --command="tail -f easylm/experiments.log"
```

### Stopping jobs

You need to run these two commands:

```sh
# Kill all processes
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME \
    --worker=all \
    --zone=$TPU_ZONE \
    --project=ai2-tpu \
    --command="sudo lsof -t /dev/accel0 | xargs sudo kill -9"
# Delete lockfiles
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME \
    --worker=all \
    --zone=$TPU_ZONE \
    --project=ai2-tpu \
    --command="sudo rm -rf /tmp/libtpu_lockfile /tmp/tpu_logs"
```
