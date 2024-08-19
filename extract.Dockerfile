FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /stage

# Install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda

# Install gsutil
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
RUN mkdir -p /usr/local/gcloud \
    && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
    && /usr/local/gcloud/google-cloud-sdk/install.sh
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

# Install EasyLM dependencies
RUN git clone https://github.com/hamishivi/EasyLM.git
RUN cd EasyLM \
    && git checkout bc241782b67bbe926e148ec9d2046d76b7ba58c8 . 
RUN conda env create -f scripts/gpu_environment.yml
ENV PATH=$CONDA_DIR/bin:$PATH
SHELL ["conda", "run", "--no-capture-output", "-n", "EasyLM", "/bin/bash", "-c"]
RUN pip install google-cloud-storage beaker-py
RUN pip install --upgrade huggingface-hub