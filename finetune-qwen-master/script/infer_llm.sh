#!/bin/bash
# 加载其他shell脚本
source ./script/source.sh $1 --retain 3 --memory 30
if [ -n "$2" ]; then
    checkpoint="$2"
else
    log_error "没有传入checkpoint参数, 请传入checkpoint参数"
    exit 1
fi

log_info "开始执行推理任务"
log_info "检查点路径: ${checkpoint}"
log_info "当前时间: $(date)"

# "${PARENT_DIR}/resource/datasets/without_instruct/MagicBrush_match.json" \

# 判断文件存在
# VAL_DATASET="${PARENT_DIR}/resource/datasets/without_instruct/$3.json"
# if [ ! -f "$VAL_DATASET" ]; then
#     log_error "测试集文件不存在: $VAL_DATASET"
#     exit 1
# fi

# 执行训练任务
python -m torch.distributed.run \
    --nproc_per_node="${nproc_per_node}" \
    --master_port ${MASTER_PORT} \
    ${PARENT_DIR}/inference_llm.py \
    --model $MODEL_NAME \
    --use_hf "true" \
    --add_expert_feat "true" \
    --add_mask_predict "true" \
    --add_cls_predict "true" \
    --template "forensic_template" \
    --system "${PARENT_DIR}/finetune/templates/system_with_reason.txt" \
    --val_dataset "${PARENT_DIR}/resource/datasets/test.json" \
    --adapters ${checkpoint} \
    --attn_impl "flash_attn" \
    --torch_dtype "bfloat16" \
    --stream "false" \
    --max_batch_size 1 \
    --infer_backend "pt" \
    --load_data_args "true"
