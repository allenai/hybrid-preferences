tpu_name=$1
echo "Stopping TPU jobs for: $tpu_name"
gcloud alpha compute tpus tpu-vm ssh $tpu_name --worker=all --zone=us-east1-d --project=ai2-tpu --command="sudo lsof -t /dev/accel0 | xargs sudo kill -9"
gcloud alpha compute tpus tpu-vm ssh $tpu_name --worker=all --zone=us-east1-d --project=ai2-tpu --command="sudo rm -rf /tmp/libtpu_lockfile /tmp/tpu_logs"