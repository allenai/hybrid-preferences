# Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback

<p align="center">
<b><a href="https://huggingface.co/datasets/allenai/multipref">🤗 Preference Dataset</a></b>
|
<b><a href="https://github.com/allenai/hybrid-preferences/tree/main/docs">📚 Documentation</a></b>
|
<b><a href="https://aclanthology.org/2025.acl-long.355/">📄 Paper</a></b>
</p>

This repository is the source code for the ACL 2025 paper, [_Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback_](https://arxiv.org/abs/2410.19133), where we introduce a **routing framework that creates hybrid preferences** with both LLM and human preference annotations to maximize performance on a given evaluation metric (e.g., RewardBench).
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

## Citation

```
@inproceedings{miranda-etal-2025-hybrid,
    title = "Hybrid Preferences: Learning to Route Instances for Human vs. {AI} Feedback",
    author = "Miranda, Lester James Validad  and
      Wang, Yizhong  and
      Elazar, Yanai  and
      Kumar, Sachin  and
      Pyatkin, Valentina  and
      Brahman, Faeze  and
      Smith, Noah A.  and
      Hajishirzi, Hannaneh  and
      Dasigi, Pradeep",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.355/",
    pages = "7162--7200",
    ISBN = "979-8-89176-251-0",
    abstract = "Learning from human feedback has enabled the alignment of language models (LMs) with human preferences. However, collecting human preferences is expensive and time-consuming, with highly variable annotation quality. An appealing alternative is to distill preferences from LMs as a source of synthetic annotations, offering a cost-effective and scalable alternative, albeit susceptible to other biases and errors. In this work, we introduce HyPER, a Hybrid Preference routER that defers an annotation to either humans or LMs, achieving better annotation quality while reducing the cost of human-only annotation. We formulate this as an optimization problem: given a preference dataset and an evaluation metric, we (1) train a performance prediction model (PPM) to predict a reward model{'}s (RM) performance on an arbitrary combination of human and LM annotations and (2) employ a routing strategy that selects a combination that maximizes predicted performance. We train the PPM on MultiPref, a new preference dataset with 10K instances paired with human and LM labels. We show that the selected hybrid mixture of synthetic and direct human preferences using HyPER achieves better RM performance compared to using either one exclusively by 7-13{\%} on RewardBench and generalizes across unseen preference datasets and other base models. We also observe the same trend in other benchmarks using Best-of-N reranking, where the hybrid mix has 2-3{\%} better performance. Finally, we analyze features from HyPER and find that prompts with moderate safety concerns or complexity benefit the most from human feedback."
}
```
