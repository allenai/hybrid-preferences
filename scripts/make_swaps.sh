sampling=$1
model=$2

python3 -m scripts.sample_best_subset \
    --input_path data/helpsteer2_all_features/features.jsonl \
    --output_dir data/helpsteer2_best_mixes_simulated_$model/ \
    --model_path data/helpsteer2_${model}_model/model.pkl \
    --budget 0.25 0.50 0.75 \
    --response_a_col response_a --response_b_col response_b \
    --sampling_method $sampling


python3 -m scripts.sample_best_subset \
    --input_path data/alpacafarm_all_features/features.jsonl \
    --output_dir data/alpacafarm_best_mixes_simulated_$model/ \
    --model_path data/alpacafarm_${model}_model/model.pkl \
    --budget 0.25 0.50 0.75 \
    --response_a_col response_a --response_b_col response_b \
    --sampling_method $sampling


python3 -m scripts.sample_best_subset \
    --input_path data/multipref_all_features/features.jsonl \
    --output_dir data/multipref_best_mixes_simulated_$model/ \
    --model_path data/multipref_${model}_model/model.pkl \
    --budget 0.25 0.50 0.75 \
    --response_a_col response_a --response_b_col response_b \
    --sampling_method $sampling

python3 -m scripts.sample_best_subset \
    --input_path data/chatarena_all_features/features.jsonl \
    --output_dir data/chatarena_best_mixes_simulated_$model/ \
    --model_path data/chatarena_${model}_model/model.pkl \
    --budget 0.25 0.50 0.75 \
    --response_a_col response_a --response_b_col response_b \
    --sampling_method $sampling
