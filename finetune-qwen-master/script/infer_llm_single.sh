#!/bin/bash
# 加载其他shell脚本
source ./script/source.sh $1 --retain 1 --memory 20
if [ -n "$2" ]; then
    checkpoint="$2"
else
    log_error "没有传入checkpoint参数, 请传入checkpoint参数"
    exit 1
fi

log_info "开始执行推理任务"
log_info "检查点路径: ${checkpoint}"
log_info "当前时间: $(date)"

# 执行训练任务
python ${PARENT_DIR}/inference_llm.py \
  --model $MODEL_NAME \
  --use_hf "true" \
  --template "forensic_template" \
  --system "${PARENT_DIR}/finetune/templates/system_with_reason.txt" \
  --val_dataset "${PARENT_DIR}/resource/datasets/without_instruct/MagicBrush.json" \
  --adapters ${checkpoint} \
  --attn_impl "flash_attn" \
  --torch_dtype "bfloat16" \
  --stream "false" \
  --max_new_tokens "2048" \
  --max_batch_size 8 \
  --infer_backend "pt" \
  --load_data_args "true"