# human-pref-datamodel

Install the dependencies within your Python environment:

```sh
python -m venv venv
venv/bin/source activate
pip install -r requirements.txt
```

## Running the extractor

First you need to pass an input JSONL file that contains the following fields:

**Required fields for input file**
- **prompt_hash**: unique ID for each prompt.
- **text**, **response_a**, **response_b**: fields containing the preference triple.
- **pref_human**: field containing the human preference (must be `A-is-clearly-better`, `Tie`, `B-is-clearly-better`)
- **pref_gpt4**: field containing the GPT-4 preference (must be `A-is-clearly-better`, `Tie`, `B-is-clearly-better`)

```sh
# Use 'multi' if you want to create multiple combinations
# Use 'single' if you just want to extract a single feature
python -m scripts.apply_data_model multi \
    --input_path path/to/dataset.jsonl \ 
    --output_dir path/to/output/dir \  
    --num_instances 7000 \
    --threshold 1 \
    --keep_features_dir path/to/features/dir \
    --append_to_experiments_file path/to/experiments.txt \
    --random_seed 42 \
    --n_sample_combinations 120 \
    --ignore_list random
```

**Optional Parameters**
- **threshold**: percentage of active features in order for an instance to be swapped with human preferences (0 to 1).
- **num_instances**: after getting the features, sample the instances to this number. Useful to match counts in previous experiments.
- **append_to_experiments_file**: create a TXT file that stores the names of the datasets created per feature combination. **Useful for passing jobs to the TPU**.
- **n_sample_combinations**: number of feature combinations to sample. Right now this can balloon if you have many features, so this parameter will randomly sample from that space.
- **ignore_list**: features to ignore.

Ideally, you should run this command in Beaker since some features require GPU (calculating BERT score, etc.). 
It's also more convenient as the output datasets are already stored in the cloud.
For reference, please check [this YAML file](https://github.com/allenai/human-pref-datamodel/blob/main/beaker/get_features.yml). Here's an [example job](https://beaker.org/ex/01J59HBTHJ8464EA4HEK0AF3NY/tasks/01J59HBTHRKYWJW9G5AKDDARK8/job/01J59HBVHMAH402NR70ST2HMBB).

## Submitting TPU jobs

You need to upload the JSONL datasets in Google Cloud Storage.
In addition, you also need to get the `experiments.txt` (the file created when you pass something to `--append_to_experiments_file`) file as this automatically lists all experiments we want to run in the TPU.
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

🐛 **Known bug**: sometimes, connecting to the wandb server fails in a node and that would cause the whole training process to hang indefinitely.
When that happens, just kill the process and remove the `--log_to_wandb` flag.
Your training should proceed, but it will not log to wandb anymore.

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