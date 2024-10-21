sampling=$1
model=$2
feat_source=$3

python3 -m scripts.sample_best_subset \
    --input_path data/helpsteer2_all_features/features.jsonl \
    --output_dir data/helpsteer2_best_mixes_${sampling}_${model}/ \
    --model_path data/${feat_source}_${model}_model/model.pkl \
    --budget 0.25 0.50 0.75 \
    --response_a_col response_a --response_b_col response_b \
    --sampling_method $sampling


python3 -m scripts.sample_best_subset \
    --input_path data/alpacafarm_all_features/features.jsonl \
    --output_dir data/alpacafarm_best_mixes_${sampling}_${model}/ \
    --model_path data/${feat_source}_${model}_model/model.pkl \
    --budget 0.25 0.50 0.75 \
    --response_a_col response_a --response_b_col response_b \
    --sampling_method $sampling


python3 -m scripts.sample_best_subset \
    --input_path data/multipref_all_features/features.jsonl \
    --output_dir data/multipref_best_mixes_${sampling}_${model}/ \
    --model_path data/${feat_source}_${model}_model/model.pkl \
    --budget 0.25 0.50 0.75 \
    --response_a_col response_a --response_b_col response_b \
    --sampling_method $sampling

python3 -m scripts.sample_best_subset \
    --input_path data/chatarena_all_features/features.jsonl \
    --output_dir data/chatarena_best_mixes_${sampling}_${model}/ \
    --model_path data/${feat_source}_${model}_model/model.pkl \
    --budget 0.25 0.50 0.75 \
    --response_a_col response_a --response_b_col response_b \
    --sampling_method $sampling

python3 -m scripts.sample_best_subset \
    --input_path data/multiprefEXP_all_features/features.jsonl \
    --output_dir data/multiprefEXP_best_mixes_${sampling}_${model}/ \
    --model_path data/${feat_source}_${model}_model/model.pkl \
    --budget 0.25 0.50 0.75 \
    --response_a_col response_a --response_b_col response_b \
    --sampling_method $sampling

python3 -m scripts.sample_best_subset \
    --input_path data/multiprefNOR_all_features/features.jsonl \
    --output_dir data/multiprefNOR_best_mixes_${sampling}_${model}/ \
    --model_path data/${feat_source}_${model}_model/model.pkl \
    --budget 0.25 0.50 0.75 \
    --response_a_col response_a --response_b_col response_b \
    --sampling_method $sampling

python3 -m scripts.sample_best_subset \
    --input_path data/hs2p_all_features/features.jsonl \
    --output_dir data/hs2p_best_mixes_${sampling}_${model}/ \
    --model_path data/${feat_source}_${model}_model/model.pkl \
    --budget 0.25 0.50 0.75 \
    --response_a_col response_a --response_b_col response_b \
    --sampling_method $sampling