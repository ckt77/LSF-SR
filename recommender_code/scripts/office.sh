#!/bin/bash

set -euo pipefail

# Assume this script is run under recommender_code/scripts/
cd ..

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SEEDS=(2023 2024 2025 2026 2027)

for SEED in "${SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python main.py \
        --beta_end 0.2 \
        --beta_start 0.01 \
        --cyclical_period 30 \
        --cyclical_ratio 0.8 \
        --dataset_file '../build_datasets_and_prompts/data/Office_Products/Office_Products.txt' \
        --experiment_name radial_3 \
        --flow_init_scale 0.001 \
        --flow_type radial \
        --item_semantic_emb_file '../build_datasets_and_prompts/data/Office_Products/office_products_item_semantic_embeddings_from_prompt.pt' \
        --lr_cosine_period 30 \
        --num_flows 3 \
        --seed ${SEED} \
        --stopping_step 60 \
        --valid_metric MRR@20
done