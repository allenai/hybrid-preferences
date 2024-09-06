#!/bin/bash


for seed in 42 10010 21; do
    mkdir -p data/baselines/helpsteer2
    python3 -m scripts.prepare_baselines \
        --output_dir data/baselines/helpsteer2/ \
        --prefix helpsteer2 \
        --id_col prompt_hash \
        --input_path data/human_vs_gpt4/helpsteer2_human_vs_gpt4_weighted_for_llama.jsonl \
        --prompt_col text \
        --completion_a_col response_a \
        --completion_b_col response_b \
        --seed $seed

    mkdir -p data/baselines/multipref
    python3 -m scripts.prepare_baselines \
        --output_dir data/baselines/multipref/ \
        --prefix multipref \
        --id_col comparison_id \
        --input_path data/human_vs_gpt4/multipref_human_vs_gpt4_overall.jsonl \
        --prompt_col text \
        --completion_a_col response_a \
        --completion_b_col response_b \
        --seed $seed

    mkdir -p data/baselines/alpacafarm
    python3 -m scripts.prepare_baselines \
        --output_dir data/baselines/alpacafarm/ \
        --prefix multipref \
        --id_col comparison_id \
        --input_path data/human_vs_gpt4/alpacafarm_human_vs_gpt4_alpacaeval.jsonl \
        --prompt_col text \
        --completion_a_col response_a \
        --completion_b_col response_b \
        --seed $seed
done
