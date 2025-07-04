#!/bin/bash
# 加载其他shell脚本
source ./script/source.sh $1 --retain 1 --memory 20 -b 4

# 执行训练任务
log_info "=====开始训练任务===="
log_info "执行模型:$MODEL_NAME"
log_info "输出目录:$OUTPUT_DIR"
log_info "日志目录:$LOG_DIR"
log_info "训练时间戳:$(date +%Y-%m-%d_%H-%M-%S)"
log_info "===================="

# 断点续训练
# --gradient_checkpointing "true" \
# --val_dataset "${PARENT_DIR}/resource/datasets/without_instruct/MagicBrush.json" \
# --dataset "${PARENT_DIR}/resource/datasets/with_instruct/CASIAv2_fake_with_instruct.json"\
# 似乎SAM的输出和cls输出相结合会导致性能下降

python -m torch.distributed.run \
    --nproc_per_node="${nproc_per_node}" \
    --master_port ${MASTER_PORT} \
    ${PARENT_DIR}/sft_llm.py \
    --model $MODEL_NAME \
    --use_hf "true" \
    --template "forensic_template" \
    --system "${PARENT_DIR}/finetune/templates/system_with_reason.txt" \
    --add_expert_feat "false" \
    --add_mask_predict "false" \
    --add_cls_predict "false" \
    --cls_weight "1" \
    --bce_weight "0.5" \
    --dice_weight "0.5" \
    --trufor_config_path "${PARENT_DIR}/config/trufor.yaml" \
    --dataset "${PARENT_DIR}/resource/datasets/with_instruct/CASIAv2_fake_with_instruct.json" "${PARENT_DIR}/resource/datasets/with_instruct/CASIAv2_real_with_instruct.json"\
    --split_dataset_ratio "0.1" \
    --attn_impl "flash_attn" \
    --train_type "lora" \
    --torch_dtype "bfloat16" \
    --data_seed 42 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_train_batch_size} \
    --modules_to_save "embed_tokens" "lm_head" "cls_head" "mask_head"\
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 8 \
    --target_modules "all-linear" \
    --freeze_vit true \
    --gradient_accumulation_steps ${total_batch_size} \
    --num_train_epochs 20 \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --logging_steps 2 \
    --max_length 2048 \
    --output_dir ${OUTPUT_DIR} \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --report_to "tensorboard" 2>&1