#!/bin/bash


for random_seed in 42 10010 21; do
    mkdir -p data/baselines/helpsteer2
    python3 -m scripts.prepare_baselines \
        --output_dir data/baselines/helpsteer2/ \
        --prefix helpsteer2 \
        --id_col prompt_hash \
        --input_path data/human_vs_gpt4/helpsteer2_human_vs_gpt4_weighted_for_llama.jsonl \
        --prompt_col text \
        --completion_a_col response_a \
        --completion_b_col response_b \
        --random_seed $random_seed

    mkdir -p data/baselines/multipref
    python3 -m scripts.prepare_baselines \
        --output_dir data/baselines/multipref/ \
        --prefix multipref \
        --id_col comparison_id \
        --input_path data/human_vs_gpt4/multipref_human_vs_gpt4_overall.jsonl \
        --prompt_col text \
        --completion_a_col response_a \
        --completion_b_col response_b \
        --random_seed $random_seed

    mkdir -p data/baselines/alpacafarm
    python3 -m scripts.prepare_baselines \
        --output_dir data/baselines/alpacafarm/ \
        --prefix alpacafarm \
        --id_col prompt_hash \
        --input_path data/human_vs_gpt4/alpacafarm_human_vs_gpt4_alpacaeval.jsonl \
        --prompt_col text \
        --completion_a_col response_a \
        --completion_b_col response_b \
        --random_seed $random_seed

    mkdir -p data/baselines/chatarena
    python3 -m scripts.prepare_baselines \
        --output_dir data/baselines/chatarena/ \
        --prefix chatarena \
        --id_col prompt_hash \
        --input_path data/human_vs_gpt4/chatarena_human_vs_gpt4_alpacaeval.jsonl \
        --prompt_col text \
        --completion_a_col response_a \
        --completion_b_col response_b \
        --random_seed $random_seed

    mkdir -p data/baselines/mphs2
    python3 -m scripts.prepare_baselines \
        --output_dir data/baselines/mphs2 \
        --prefix mphs2 \
        --id_col id \
        --input_path data/mphs2_all_features/features.jsonl \
        --prompt_col text \
        --completion_a_col response_a \
        --completion_b_col response_b \
        --random_seed $random_seed \
        --num_instances 17000

    mkdir -p data/baselines/multiprefNOR
    python3 -m scripts.prepare_baselines \
        --output_dir data/baselines/multiprefNOR \
        --prefix multiprefNOR \
        --id_col id \
        --input_path data/multiprefNOR_all_features/features.jsonl \
        --prompt_col text \
        --completion_a_col response_a \
        --completion_b_col response_b \
        --random_seed $random_seed \
        --num_instances 7000

    mkdir -p data/baselines/multiprefEXP
    python3 -m scripts.prepare_baselines \
        --output_dir data/baselines/multiprefEXP \
        --prefix multiprefEXP \
        --id_col id \
        --input_path data/multiprefEXP_all_features/features.jsonl \
        --prompt_col text \
        --completion_a_col response_a \
        --completion_b_col response_b \
        --random_seed $random_seed \
        --num_instances 7000

    mkdir -p data/baselines/hs2p
    python3 -m scripts.prepare_baselines \
        --output_dir data/baselines/hs2p \
        --prefix hs2p \
        --id_col id \
        --input_path data/hs2p_all_features/features.jsonl \
        --prompt_col text \
        --completion_a_col response_a \
        --completion_b_col response_b \
        --random_seed $random_seed \
        --num_instances 7000
done
