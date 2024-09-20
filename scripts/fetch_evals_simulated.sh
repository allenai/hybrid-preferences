python3 scripts/fetch_evals_rewardbench.py \
    --output_path data/multipref-counts-runs-simulated-linear.csv \
    --experiment_prefix rm-eval-multipref-count \
    --feature_counts_dir data/multipref_best_mixes_simulated_linear/counts/ \
    --dataset_total_size 10461

python3 scripts/fetch_evals_rewardbench.py \
    --output_path data/multipref-counts-runs-simulated-quadratic.csv \
    --experiment_prefix rm-eval-multipref-count \
    --feature_counts_dir data/multipref_best_mixes_simulated_quadratic/counts/ \
    --dataset_total_size 10461


python3 scripts/fetch_evals_rewardbench.py \
    --output_path data/helpsteer2-counts-runs-simulated-linear.csv \
    --experiment_prefix rm-eval-helpsteer2-count \
    --feature_counts_dir data/helpsteer2_best_mixes_simulated_linear/counts/ \
    --dataset_total_size 10461

python3 scripts/fetch_evals_rewardbench.py \
    --output_path data/helpsteer2-counts-runs-simulated-quadratic.csv \
    --experiment_prefix rm-eval-helpsteer2-count \
    --feature_counts_dir data/helpsteer2_best_mixes_simulated_quadratic/counts/ \
    --dataset_total_size 10160

python3 scripts/fetch_evals_rewardbench.py \
    --output_path data/alpacafarm-counts-runs-simulated-linear.csv \
    --experiment_prefix rm-eval-alpacafarm-count \
    --feature_counts_dir data/alpacafarm_best_mixes_simulated_linear/counts/ 

python3 scripts/fetch_evals_rewardbench.py \
    --output_path data/alpacafarm-counts-runs-simulated-quadratic.csv \
    --experiment_prefix rm-eval-alpacafarm-count \
    --feature_counts_dir data/alpacafarm_best_mixes_simulated_quadratic/counts/ 

python3 scripts/fetch_evals_rewardbench.py \
    --output_path data/chatarena-counts-runs-simulated-linear.csv \
    --experiment_prefix rm-eval-chatarena-count \
    --feature_counts_dir data/chatarena_best_mixes_simulated_linear/counts/ 

python3 scripts/fetch_evals_rewardbench.py \
    --output_path data/chatarena-counts-runs-simulated-quadratic.csv \
    --experiment_prefix rm-eval-chatarena-count \
    --feature_counts_dir data/chatarena_best_mixes_simulated_quadratic/counts/ 