REVISION=(
    "human_datamodel_counts_7000_ID__03242bb1814b48978f32cfa8090308e2__SWAPS_2169"
    "human_datamodel_counts_7000_ID__12210931507643a7b7eed73e878eae1f__SWAPS_6507"
    "human_datamodel_counts_7000_ID__3e2014f53dba499cbdac39fda22e24a5__SWAPS_4338"
    "human_datamodel_counts_7000_ID__5d3de9a4893b4a0ab79234af754954e1__SWAPS_2169"
    "human_datamodel_counts_7000_ID__5dc9108715934d989e0080d9111506bf__SWAPS_6507"
    "human_datamodel_counts_7000_ID__9be2ce559efd4a59ba46478f3f9ed502__SWAPS_4338"
    "human_datamodel_counts_7000_ID__a00a49d8a9404752bba39a60196dc650__SWAPS_2169"
    "human_datamodel_counts_7000_ID__a2c10e356fa84d449b9849c911b50a72__SWAPS_6507"
    "human_datamodel_counts_7000_ID__c4ee928ab76b483d9d9b2c8b3c1ef9c3__SWAPS_2169"
    "human_datamodel_counts_7000_ID__c8d950699ce94fbc86b71eef9e1f5b52__SWAPS_4338"
    "human_datamodel_counts_7000_ID__ca3ec66db56a41d48b94cfc1f2242657__SWAPS_2169"
    "human_datamodel_counts_7000_ID__e6316c853a4d4974ab0eaa5c27571d5c__SWAPS_2169"
    "hs2p_human_SWAPS_6766_SEED_42"
    "hs2p_human_75_SWAPS_4938_SEED_42"
    "hs2p_human_50_SWAPS_3060_SEED_42"
    "hs2p_human_25_SWAPS_1433_SEED_42"
    "hs2p_gpt4_SWAPS_0_SEED_42"
    "hs2p_random_SWAPS_3031_SEED_42"
    "hs2p_human_SWAPS_6766_SEED_10010"
    "hs2p_human_75_SWAPS_4899_SEED_10010"
    "hs2p_human_50_SWAPS_3062_SEED_10010"
    "hs2p_human_25_SWAPS_1467_SEED_10010"
    "hs2p_gpt4_SWAPS_0_SEED_10010"
    "hs2p_random_SWAPS_3050_SEED_10010"
)

mkdir -p rewardbench_results/
for split in "${REVISION[@]}"; do
    rewardbench --model ljvmiranda921/helpsteer2-qwen-rms \
        --revision $split \
        --chat_template tulu \
        --save_all \
        --output_dir rewardbench_results/
    mv rewardbench_results/ljvmiranda921/helpsteer2-qwen-rms.json $split.json
done