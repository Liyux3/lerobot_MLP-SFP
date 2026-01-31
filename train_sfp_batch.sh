#!/bin/bash

DATASET="/home/yuxianli/.cache/huggingface/lerobot/yuxianli/pusht_real_merged"
STEPS=50000
SAVE_FREQ=10000

mkdir -p ./logs

train_model() {
    local net_type=$1
    local version=$2
    local gpu=$3
    
    echo "[$(date '+%H:%M:%S')] Starting $net_type ($version) on GPU $gpu..."
    
    CUDA_VISIBLE_DEVICES=$gpu lerobot-train \
        --dataset.repo_id=$DATASET \
        --policy.type=sfp \
        --policy.velocity_net_type=$net_type \
        --policy.device=cuda:0 \
        --policy.repo_id=yuxianli/pusht_sfp_$version \
        --policy.push_to_hub=false \
        --output_dir=./outputs/pusht_sfp_$version \
        --batch_size=32 \
        --steps=$STEPS \
        --save_freq=$SAVE_FREQ \
        --wandb.enable=true \
        --wandb.project=pusht_fyp \
        > ./logs/sfp_${version}.log 2>&1
    
    echo "[$(date '+%H:%M:%S')] Finished $version"
}

# v1 already done (bottleneck_skip fixed)

# Round 1: parallel on both GPUs
train_model "bottleneck_skip_auto" "v2_auto" 0 &
train_model "unet_dp" "v3_unet_dp" 1 &
wait

# Round 2: parallel
train_model "pure_mlp" "v4_pure" 0 &
train_model "cond_every_layer" "v5_cond" 1 &
wait

# Round 3: parallel
train_model "cond_residual" "v6_residual" 0 &
train_model "unet" "v7_unet" 1 &
wait

echo "[$(date '+%H:%M:%S')] All training complete!"
