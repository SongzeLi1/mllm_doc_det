#!/bin/bash
# 脚本用于设置GPU训练环境和计算训练参数
set -e # 遇到错误立即退出

# ANSI颜色代码
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

# 显示带颜色的信息函数
log_info() {
    printf "${BLUE}[INFO]${NC} %s\n" "$1"
}

log_success() {
    printf "${GREEN}[SUCCESS]${NC} %s\n" "$1"
}

log_warning() {
    printf "${YELLOW}[WARNING]${NC} %s\n" "$1"
}

log_error() {
    printf "${RED}[ERROR]${NC} %s\n" "$1" >&2
    exit 1
}

# 帮助信息
show_help() {
    cat <<EOF
用法: $0 <模型参数> [选项]
模型参数:
  3b      对应模型: Qwen/Qwen2.5-VL-3B-Instruct
  7b      对应模型: Qwen/Qwen2.5-VL-7B-Instruct
  32b     对应模型: Qwen/Qwen2.5-VL-32B-Instruct
  其他参数    对应的参数模型
选项:
  -m, --memory <GB>       最小GPU显存要求(GB) (默认: 15)
  -b, --batch-size <N>    每设备批次大小 (默认: 2)
  -r, --retain            保留的GPU数量
  -h, --help              显示此帮助信息
EOF
}

# 初始化参数
gpu_memory=15
retain=0
per_device_train_batch_size=1
manual_gpu=""     # 新增：手动指定GPU列表

# 模型参数判断及设置，要求第一个参数必须传入模型信息
if [ -z "$1" ]; then
    log_error "请提供模型参数（3b、7b、32b）"
fi

DEEPSPEED_CONFIG="zero2_offload"
case "$1" in
"3b")
    MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
    ;;
"7b")
    MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
    ;;
"32b")
    MODEL_NAME="Qwen/Qwen2.5-VL-32B-Instruct"
    DEEPSPEED_CONFIG="zero3"
    ;;
*)
    # 额外的变量即为模型的名称
    MODEL_NAME="$1"
    ;;
esac

# 移除模型参数，以便后续参数解析
shift

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
    -m | --memory)
        gpu_memory="$2"
        shift 2
        ;;
    -b | --batch-size)
        per_device_train_batch_size="$2"
        shift 2
        ;;
    -r | --retain)
        retain="$2"
        shift 2
        ;;
    -g)
        manual_gpu="$2"
        shift 2
        ;;
    -h | --help)
        show_help
        exit 0
        ;;
    *)
        if [[ "$1" =~ ^[0-9]+$ ]] && [ -z "${gpu_memory_set:-}" ]; then
            gpu_memory="$1"
            gpu_memory_set=true
            shift
        elif [[ "$1" =~ ^[0-9]+$ ]] && [ -z "${bs_set:-}" ]; then
            per_device_train_batch_size="$1"
            bs_set=true
            shift
        elif [[ "$1" =~ ^[0-9]+$ ]] && [ -z "${gs_set:-}" ]; then
            grad_acc_steps="$1"
            gs_set=true
            shift
        else
            log_error "未知参数: $1"
        fi
        ;;
    esac
done

# 检查参数有效性
if ! [[ "$gpu_memory" =~ ^[0-9]+$ ]]; then
    log_error "GPU显存必须是整数"
fi
if ! [[ "$retain" =~ ^[0-9]+$ ]]; then
    log_error "保留的GPU数量必须是整数"
fi
if ! [[ "$per_device_train_batch_size" =~ ^[0-9]+$ ]]; then
    log_error "批次大小必须是整数"
fi

# 设置GPU环境
setup_gpu_environment() {
    if [ -n "$manual_gpu" ]; then
        log_info "手动指定GPU: ${manual_gpu}"
        export CUDA_VISIBLE_DEVICES="$manual_gpu"
    else
        log_info "正在查询可用GPU (最小显存: ${gpu_memory}GB), 保留GPU数量 ${retain}个..."
        result=$(python script/load_gpu.py --memory "$gpu_memory" --retain "$retain")
        echo "$result"
        log_success "获取可用的GPU: $result"
        export CUDA_VISIBLE_DEVICES="$result"
    fi

    # 获取GPU数量
    nproc_per_node=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
    log_info "可用GPU数量: $nproc_per_node"
    log_info "每个GPU的batch_size: ${per_device_train_batch_size}"

    # 计算总的batch size
    if [ "$nproc_per_node" -gt 0 ]; then
        total_batch_size=$((per_device_train_batch_size * nproc_per_node))
    else
        total_batch_size=$per_device_train_batch_size
    fi
    # 如果总的batch size超过32，则调整为32
    if [ "$total_batch_size" -gt 32 ]; then
        total_batch_size=32
        log_warning "总的梯度累计batch_size超过32，已调整为32"
    fi
    log_info "总的梯度累计batch_size: $total_batch_size"
}

# 准备输出目录
prepare_output() {
    timestamp=$(date +"%Y-%m-%d_%H-%M")
    model_base=$(echo "$MODEL_NAME" | sed 's|.*/||')
    OUTPUT_DIR="./output"
    mkdir -p "${OUTPUT_DIR}"

    # 创建日志目录
    LOG_DIR="${OUTPUT_DIR}/logs"
    mkdir -p "${LOG_DIR}"
}


# 主函数
main() {
    log_info "开始设置训练环境..."
    setup_gpu_environment
    prepare_output
    log_success "环境设置完成!"
}

# 执行主函数
main

# 导出函数方便子 shell 调用
export -f log_info log_success log_warning log_error
# 导出训练需要的环境变量
export per_device_train_batch_size
export grad_acc_steps
export total_batch_size
export nproc_per_node
export MODEL_NAME
export DEEPSPEED_CONFIG
export OUTPUT_DIR
export LOG_DIR
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PARENT_DIR="$(dirname "${SCRIPT_DIR}")"

export MASTER_PORT=29500
# 检查端口是否占用，若占用则MASTER_PORT加2
if lsof -iTCP:${MASTER_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    log_info "端口${MASTER_PORT}被占用，切换到端口$(($MASTER_PORT + 2))"
    MASTER_PORT=$(($MASTER_PORT + 2))
fi
