for i in 256 512 1024 2048 4096 8192; do
    python3 -m scripts.sample_best_subset \
        --input_path data/hs2p_all_features/features.jsonl \
        --output_dir data/hs2p_best_mixes_optimal_simulated_$i \
        --model_path data/multipref_quadratic_model/model.pkl \
        --sampling_method optimal_simulated \
        --budgets 0.25 0.50 0.75 \
        --n_simulations $i \
        --response_a_col response_a \
        --response_b_col response_b 
done
