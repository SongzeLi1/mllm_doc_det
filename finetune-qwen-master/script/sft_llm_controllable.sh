#!/bin/bash
# 加载其他shell脚本
source ./script/source.sh $1 --retain 1 --memory 30

# 执行训练任务
log_info "=====开始训练任务===="
log_info "执行模型:$MODEL_NAME"
log_info "输出目录:$OUTPUT_DIR"
log_info "日志目录:$LOG_DIR"
log_info "训练时间戳:$(date +%Y-%m-%d_%H-%M-%S)"
log_info "===================="

# 验证 MODEL_NAME 是否在允许列表中
allowed_models=(
  "Qwen/Qwen2.5-VL-3B-Instruct"
  "Qwen/Qwen2.5-VL-7B-Instruct"
  "Qwen/Qwen2.5-VL-32B-Instruct"
)
if [[ ! " ${allowed_models[*]} " =~ " ${MODEL_NAME} " ]]; then
  echo "Error: MODEL_NAME 必须是以下之一：${allowed_models[*]}"
  exit 1
fi

# 断点续训练
# --gradient_checkpointing "true" \
# --val_dataset "${PARENT_DIR}/resource/datasets/without_instruct/MagicBrush.json" \
# --dataset "${PARENT_DIR}/resource/datasets/with_instruct/CASIAv2_fake_with_instruct.json"\
# --resume_from_checkpoint "output/Qwen2.5-VL-32B-Instruct_2025-04-10_08-14/v0-20250410-081459/checkpoint-957" \
# 似乎SAM的输出和cls输出相结合会导致性能下降
# --resume_from_checkpoint "output/Qwen2.5-VL-7B-Instruct_sam2.1-hiera-base-plus_3box_MagicBrush_match_CASIAv2_20250515_1514_box3_pos0_cls_mask_expert/v3-20250522-071143/checkpoint-4584" \

# 执行训练任务
log_info "=====开始训练任务===="
python -m torch.distributed.run \
  --nproc_per_node="${nproc_per_node}" \
  --master_port ${MASTER_PORT} \
  ${PARENT_DIR}/sft_llm.py \
  --model $MODEL_NAME \
  --use_hf "true" \
  --template "controllable_template" \
  --system "${PARENT_DIR}/finetune/templates/system_with_reason.txt" \
  --add_expert_feat "true" \
  --add_mask_predict "true" \
  --add_cls_predict "true" \
  --sam_model_path "/home/yuyangxin/data/finetune-qwen/sam_output/sam2.1-hiera-base-plus_3box_MIML_Part1_fake_CASIAv2_real-7437_20250430_0919/checkpoints/Epoch010.pt" \
  --token_pos "0" \
  --cls_weight "1" \
  --bce_weight "0.5" \
  --dice_weight "0.5" \
  --dataset "${PARENT_DIR}/resource/datasets/with_instruct/FragFake_train_hard.json" \
  --split_dataset_ratio "0.1" \
  --attn_impl "flash_attn" \
  --train_type "lora" \
  --torch_dtype "bfloat16" \
  --data_seed 42 \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --per_device_eval_batch_size ${per_device_train_batch_size} \
  --modules_to_save "embed_tokens" "lm_head" "cls_head" "mask_head" \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules "all-linear" \
  --freeze_vit true \
  --gradient_accumulation_steps ${total_batch_size} \
  --num_train_epochs 10 \
  --save_strategy "epoch" \
  --save_total_limit 10 \
  --logging_steps 2 \
  --max_length 2048 \
  --output_dir ${OUTPUT_DIR} \
  --warmup_ratio 0.1 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  --deepspeed ${DEEPSPEED_CONFIG} \
  --dataset_type "controllable" \
  --report_to "tensorboard" 2>&1
