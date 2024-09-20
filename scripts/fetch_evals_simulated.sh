sampling=$1
model=$2
dataset=$3

python3 scripts/fetch_evals_rewardbench.py \
    --output_path data/${dataset}-counts-runs-${sampling}-${model}.csv \
    --experiment_prefix rm-eval-${dataset}-count \
    --feature_counts_dir data/${dataset}_best_mixes_${sampling}_${model}/counts/ 