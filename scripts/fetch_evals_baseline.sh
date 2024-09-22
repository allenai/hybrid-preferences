for dataset in helpsteer2 multipref alpacafarm chatarena; do
    python3 scripts/fetch_evals_rewardbench.py \
        --output_path data/${dataset}-baseline-runs.csv \
        --experiment_prefix rm-eval-baseline-${dataset} \
        --feature_counts_dir data/baselines/${dataset}/ 
done