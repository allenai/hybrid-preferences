FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install \
    numpy \
    scipy \
    matplotlib \
    seaborn \
    jupyter \
    tqdm \
    sentencepiece \
    jax==0.4.14 \
    jaxlib==0.4.14 \ 
    flax==0.7.0 \
    optax==0.1.7 \
    distrax==0.1.4 \
    chex==0.1.82 \
    transformers==4.43.1 \
    torch==2.0.1 \
    datasets==2.14.2 \
    einops \
    tensorflow==2.11.1 \
    dill \
    absl-py \
    wandb \
    ml_collections \
    gcsfs \
    requests \
    jupyter_http_over_ws \
    lm-eval \
    mlxu==0.1.11 \
    pydantic \
    fastapi \
    uvicorn \
    google-cloud-storage \
    beaker-py \
    huggingface-hub \
    gradio

RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz | tar -C /usr/local -xzf - && \
    /usr/local/google-cloud-sdk/install.sh --quiet 

ENV PATH="/usr/local/google-cloud-sdk/bin:${PATH}"

# Clone EasyLM repository
RUN git clone https://github.com/hamishivi/EasyLM.git  . && \
    # git checkout bc241782b67bbe926e148ec9d2046d76b7ba58c8
    git checkout dbf2212c1775b2762f7108d62c8c8b01b52ea4aa

COPY ai2-allennlp-79f5e3a8e95a.json /root/.config/gcloud/application_default_credentials.json
# Set environment variable for Google Cloud
ENV GOOGLE_APPLICATION_CREDENTIALS="/root/.config/gcloud/application_default_credentials.json"
# Gcloud setup
RUN gcloud auth activate-service-account --key-file=/root/.config/gcloud/application_default_credentials.json && \
    gcloud config set project ai2-allennlp

# # Copy tokenizer model
RUN gsutil cp gs://hamishi-east1/easylm/llama/tokenizer.model .
COPY convert_to_hf.py .
COPY download-beaker.sh .
RUN apt-get update && \
    apt-get -y install sudo && \
    chmod +x ./download-beaker.sh && \
    ./download-beaker.sh
RUN beaker --version