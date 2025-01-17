tpu_name=$1
type=$2
zone=$3
echo "Creating TPU: $tpu_name (type: $type zone: $zone)"
while ! gcloud alpha compute tpus tpu-vm create $tpu_name --accelerator-type=$type --zone=$zone --project=ai2-tpu --version=v2-alpha; do sleep 60; done
gcloud alpha compute tpus tpu-vm ssh $tpu_name --zone=$zone --project=ai2-tpu --worker=all --command="git clone https://github.com/hamishivi/easylm.git"
gcloud alpha compute tpus tpu-vm ssh $tpu_name --zone=$zone --project=ai2-tpu --worker=all --command="cd easylm; git checkout dbf2212c1775b2762f7108d62c8c8b01b52ea4aa .; ./scripts/tpu_vm_setup.sh"
# gcloud alpha compute tpus tpu-vm ssh $tpu_name --zone=$zone --project=ai2-tpu --worker=all --command="cd easylm; git checkout bc241782b67bbe926e148ec9d2046d76b7ba58c8 .; ./scripts/tpu_vm_setup.sh"
gcloud alpha compute tpus tpu-vm ssh $tpu_name --zone=$zone --project=ai2-tpu --worker=all --command="python3 -m pip install wandb --upgrade"
gcloud alpha compute tpus tpu-vm ssh $tpu_name --zone=$zone --project=ai2-tpu --worker=all --command="python3 -m wandb login $WANDB_TOKEN"
gcloud alpha compute tpus tpu-vm ssh $tpu_name --zone=$zone --project=ai2-tpu --worker=all --command="python3 -m pip install -U 'huggingface_hub[cli]'"
gcloud alpha compute tpus tpu-vm ssh $tpu_name --zone=$zone --project=ai2-tpu --worker=all --command="echo 'export PATH=\$PATH:~/.local/bin' >> ~/.bashrc"
gcloud alpha compute tpus tpu-vm ssh $tpu_name --zone=$zone --project=ai2-tpu --worker=all --command="source ~/.bashrc"
gcloud alpha compute tpus tpu-vm ssh $tpu_name --zone=$zone --project=ai2-tpu --worker=all --command="~/.local/bin/huggingface-cli login --token $HF_TOKEN"
