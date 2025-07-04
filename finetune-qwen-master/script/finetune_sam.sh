#!/bin/bash
# 定义可选模型
VALID_MODELS=(tiny small base-plus large)

usage() {
    cat << EOF
Usage: $0 [-h] MODEL_NAME

MODEL_NAME must be one of: ${VALID_MODELS[*]}

Options:
  -h        Show this help message and exit
EOF
    exit 1
}

# 解析参数
if [[ "$1" == "-h" || -z "$1" ]]; then
    usage
fi

# 验证 MODEL_NAME 是否有效
MODEL_NAME="$1"
if [[ ! " ${VALID_MODELS[*]} " =~ " ${MODEL_NAME} " ]]; then
    echo "Error: Invalid MODEL_NAME: ${MODEL_NAME}" >&2
    usage
fi

# 加载其他 shell 脚本
source ./script/source.sh "${MODEL_NAME}" --retain 0 --memory 20 -b 32

# 执行训练任务
log_info "=====开始训练任务===="
log_info "执行模型:$MODEL_NAME"
log_info "输出目录:$OUTPUT_DIR"
log_info "日志目录:$LOG_DIR"
log_info "训练时间戳:$(date +%Y-%m-%d_%H-%M-%S)"
log_info "===================="

# "${PARENT_DIR}/resource/datasets/without_instruct/MIML_Part1_fake.json" 
# 想法: 先使用大的模型进行train
# 使用一个box进行框选(1, 3, 5)
# 相同样本进行train, 在进行测试, 结果如何?
# 
OUTPUT_DIR="${PARENT_DIR}/sam_output"
python -m torch.distributed.run \
    --nproc_per_node="${nproc_per_node}" \
    --master_port ${MASTER_PORT} \
    ${PARENT_DIR}/finetune_sam.py \
    --model_name "facebook/sam2.1-hiera-${MODEL_NAME}" \
    --dataset_paths "${PARENT_DIR}/resource/datasets/without_instruct/CASIAv2.json"\
    --ds_config "${PARENT_DIR}/configs/ds_config.json" \
    --save_dir "${OUTPUT_DIR}" \
    --batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps 3 \
    --split_ratio 0 \
    --split_box_num 3 \
    --patience 0 \
    --epochs 10 \
