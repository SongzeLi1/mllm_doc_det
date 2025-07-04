#!/bin/bash
# 加载其他shell脚本
source ./script/source.sh sam2 --retain 0 --memory 20 -b 16

log_info "=====开始训练任务===="
log_info "执行模型:$MODEL_NAME"
log_info "输出目录:$OUTPUT_DIR"
log_info "日志目录:$LOG_DIR"
log_info "训练时间戳:$(date +%Y-%m-%d_%H-%M-%S)"
log_info "===================="

OUTPUT_DIR="${PARENT_DIR}/sam_output"
python -m torch.distributed.run \
    --nproc_per_node="${nproc_per_node}" \
    --master_port ${MASTER_PORT} \
    ${PARENT_DIR}/inference_sam.py \
    --dataset_paths "${PARENT_DIR}/resource/datasets/without_instruct/MagicBrush_match.json" \
    --checkpoint $1 \
    --batch_size ${per_device_train_batch_size}