#!/bin/bash
# 加载其他shell脚本
source ./script/source.sh $1 --retain 2 --memory 20
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
python -m torch.distributed.run \
  --nproc_per_node="${nproc_per_node}" \
  --master_port ${MASTER_PORT} \
   ${PARENT_DIR}/inference_llm.py \
  --model $MODEL_NAME \
  --adapters ${checkpoint} \
  --system "${PARENT_DIR}/finetune/templates/system_with_reason.txt" \
  --template "forensic_template" \
  --stream "false" \
  --max_new_tokens "2048" \
  --max_batch_size 1 \
  --infer_backend "pt" \
  --load_data_args "true" \
  --add_expert_feat "false" \
  --add_mask_predict "false" \
  --add_cls_predict "false" \
  --torch_dtype "bfloat16" \
  --val_dataset "${PARENT_DIR}/resource/datasets/without_instruct/MagicBrush.json" \
  --use_hf "true"