for i in 256 512 1024 2048 4096 8192; do
    python3 scripts/fetch_evals_rewardbench.py \
        --output_path data/hs2p-$i-results-llama.csv \
        --experiment_prefix rm-eval-hs2p-$i-llama3
done