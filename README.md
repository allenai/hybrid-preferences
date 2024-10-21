# Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback

<p align="center">
<b><a href="https://huggingface.co/datasets/allenai/multipref">🤗 Preference Dataset</a></b>
|
<b><a href="https://github.com/allenai/hybrid-preferences/tree/main/docs">📚 Documentation</a></b>
|
<b>📄 Paper (coming soon!)</b>
</p>

This repository is the source code for the paper, [_Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback_](), where we introduce a **routing framework that creates hybrid preferences** with both LLM and human preference annotations to maximize performance on a given evaluation metric (e.g., RewardBench).
We release this codebase to improve reproducibility of our work, and to aid researchers in constructing preference datasets in their research.

<img width="2285" alt="main_figure" src="https://github.com/user-attachments/assets/3bfb7c42-ec9c-4457-9949-367dc6270269">

## Setup

Install the dependencies within your Python environment:

```sh
python -m venv venv
venv/bin/source activate
pip install -r requirements.txt
```

## Documentation

Running the full pipeline involves several steps, some might need to be run on a TPU machine.
Nevertheless, we wrote scripts to automate different parts of the pipeline.
**Please head over the [docs](https://github.com/allenai/human-pref-datamodel/tree/main/docs) directory for more information.**
