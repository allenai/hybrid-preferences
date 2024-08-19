git clone https://github.com/hamishivi/EasyLM.git
cd EasyLM
git checkout bc241782b67bbe926e148ec9d2046d76b7ba58c8 .
conda env create -f scripts/gpu_environment.yml
conda activate EasyLM
gcloud auth login
gsutil cp gs://hamishi-east1/easylm/llama/tokenizer.model .
pip install google-cloud-storage beaker-py
gcloud auth application-default login