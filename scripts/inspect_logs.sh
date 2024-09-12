tpu_name=$1
gcloud alpha compute tpus tpu-vm ssh $tpu_name --worker=all --zone=us-east1-d --project=ai2-tpu --command="tail -f easylm/experiments.log"