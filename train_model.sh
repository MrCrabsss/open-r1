#!/bin/bash

accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path /workspace/models/Qwen2.5-3B-Instruct \
    --train_data_path /workspace/open-r1/data/train.parquet \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 \
    --output_dir /workspace/open-r1/Qwen2.5-3B-Instruct-R1-Distill \
    --max_steps 10 \
    --eval_steps 5 \
    --evaluation_strategy steps