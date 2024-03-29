#!/bin/bash

MODEL="/root/tuned_models/yi34b200k_ark_v1" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="/root/yumu/Qwen1.5/data/qwen_format_long_data.jsonl"
DS_CONFIG_PATH="ds_config_zero3.json"
USE_LORA=True
Q_LORA=False

deepspeed finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 True \
    --output_dir /root/yumu/Qwen1.5/examples/sft/output_qwen \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 1 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --do_train \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 10000 \
    --lazy_preprocess True \
    --remove_unused_columns False \
    --use_lora ${USE_LORA} \
    --q_lora ${Q_LORA} \
    --gradient_checkpointing \
    --deepspeed ${DS_CONFIG_PATH}
