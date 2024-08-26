FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /stage

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git
COPY requirements.txt /stage
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_lg
RUN python -m nltk.downloader 'wordnet'
RUN python -m nltk.downloader 'punkt_tab'

# Copy all files
COPY . /stage