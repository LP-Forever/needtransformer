#!/bin/bash
# =============================================================================
# RoboVLMs Training Script (local or amlt cloud)
# =============================================================================
# RoboVLMs uses Lightning Trainer + JSON config files.
# This script wraps torchrun + main.py with a configurable JSON config.
#
# Local usage example (single GPU, KosMos backbone):
#   cd /home/v-wenhuitan/TCD/needtransformer/RoboVLMs
#   bash run_robovlms_train.sh --nproc_per_node 1 --gpu_ids 0
#
# Cloud usage (via ft_robovlms.yaml):
#   bash run_robovlms_train.sh
# =============================================================================

# --------------------------- 可配置参数 --------------------------------------

# 分布式训练
NNODES=1
NPROC_PER_NODE=8
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=6042
GPU_IDS="0,1,2,3,4,5,6,7"          # CUDA_VISIBLE_DEVICES

# 配置文件 (JSON)
CODE_DIR=${AMLT_CODE_DIR:-/home/v-wenhuitan/TCD/needtransformer/RoboVLMs}
# 默认用 KosMos (最强 backbone per README)，实机 finetune 配置
CONFIG_JSON="${CODE_DIR}/configs/custom_finetune/finetune_kosmos_cont-lstm-post_full-ft_custom_wd-0_ws-8_act-10.json"

# VLM backbone 模型路径 (模型会自动下载到 .vlms/ 下，也可指定已有路径)
# KosMos: 自动从 HF 下载 microsoft/kosmos-2-patch14-224
# 如已有本地模型，可设置 MODEL_PATH 覆盖
MODEL_PATH=""

# 数据
# OpenVLADataset: data_root_dir 下放 RLDS 格式数据
# DiskCalvinDataset: data_dir 下放 CALVIN 格式数据
DATA_ROOT_DIR="/mnt/wangxiaofa/robot_dataset/open-x-embodiment"   # 云端路径，本地需要改
DATA_MIX="custom_finetuning"

# 训练超参覆盖 (覆盖 JSON config 中的值)
BATCH_SIZE=""
MAX_STEPS=""
LEARNING_RATE=""
MODEL_LOAD_PATH=""                  # 从已有 checkpoint resume

# WandB
WANDB_PROJECT="robovlms"
WANDB_ENTITY=""

# 输出
OUTPUT_ROOT="/mnt/wangxiaofa/robovlms-output"   # 云端路径，本地需要改
LOG_ROOT="/mnt/wangxiaofa/robovlms-logs"

# --------------------------- 解析命令行参数 ----------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnodes)           NNODES="$2";           shift 2 ;;
        --nproc_per_node)   NPROC_PER_NODE="$2";   shift 2 ;;
        --node_rank)        NODE_RANK="$2";         shift 2 ;;
        --master_addr)      MASTER_ADDR="$2";       shift 2 ;;
        --master_port)      MASTER_PORT="$2";       shift 2 ;;
        --gpu_ids)          GPU_IDS="$2";           shift 2 ;;
        --config)           CONFIG_JSON="$2";       shift 2 ;;
        --model_path)       MODEL_PATH="$2";        shift 2 ;;
        --data_root_dir)    DATA_ROOT_DIR="$2";     shift 2 ;;
        --data_mix)         DATA_MIX="$2";          shift 2 ;;
        --batch_size)       BATCH_SIZE="$2";        shift 2 ;;
        --max_steps)        MAX_STEPS="$2";         shift 2 ;;
        --lr)               LEARNING_RATE="$2";     shift 2 ;;
        --model_load_path)  MODEL_LOAD_PATH="$2";   shift 2 ;;
        --output_root)      OUTPUT_ROOT="$2";       shift 2 ;;
        --log_root)         LOG_ROOT="$2";          shift 2 ;;
        --wandb_project)    WANDB_PROJECT="$2";     shift 2 ;;
        --wandb_entity)     WANDB_ENTITY="$2";      shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# --------------------------- 构建命令行参数 ----------------------------------

EXTRA_ARGS=""
if [ -n "${BATCH_SIZE}" ];    then EXTRA_ARGS="${EXTRA_ARGS} --batch_size ${BATCH_SIZE}"; fi
if [ -n "${MAX_STEPS}" ];     then EXTRA_ARGS="${EXTRA_ARGS} --trainer.max_steps ${MAX_STEPS}"; fi
if [ -n "${LEARNING_RATE}" ]; then EXTRA_ARGS="${EXTRA_ARGS} --learning_rate ${LEARNING_RATE}"; fi
if [ -n "${MODEL_LOAD_PATH}" ]; then EXTRA_ARGS="${EXTRA_ARGS} --model_load_path ${MODEL_LOAD_PATH}"; fi
if [ -n "${MODEL_PATH}" ];    then EXTRA_ARGS="${EXTRA_ARGS} --model_path ${MODEL_PATH}"; fi

# --------------------------- 打印配置 --------------------------------------

echo "============================================"
echo "  RoboVLMs Training"
echo "============================================"
echo "  Config JSON:     ${CONFIG_JSON}"
echo "  GPUs:            ${NPROC_PER_NODE} x ${NNODES} nodes"
echo "  GPU IDs:         ${GPU_IDS}"
echo "  Data mix:        ${DATA_MIX}"
echo "  Data root:       ${DATA_ROOT_DIR}"
echo "  Output root:     ${OUTPUT_ROOT}"
echo "  Log root:        ${LOG_ROOT}"
echo "  WandB:           ${WANDB_PROJECT}"
echo "  Extra args:      ${EXTRA_ARGS}"
echo "============================================"

# --------------------------- 准备 KosMos transformers patch --------------------------------------
# KosMos 需要替换 transformers 中的 modeling_kosmos2.py
# 仅在首次运行时需要
if [ ! -f "${CODE_DIR}/.vlms/kosmos-2-patch14-224/config.json" ]; then
    echo "KosMos model not found locally, will download from HF."
fi

# --------------------------- 启动训练 --------------------------------------

cd "${CODE_DIR}"

CUDA_VISIBLE_DEVICES=${GPU_IDS} \
torchrun \
  --nnodes=${NNODES} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
main.py \
  "${CONFIG_JSON}" \
  --gpus ${NPROC_PER_NODE} \
  --num_nodes ${NNODES} \
  --output_root "${OUTPUT_ROOT}" \
  --log_root "${LOG_ROOT}" \
  --train_dataset.data_root_dir "${DATA_ROOT_DIR}" \
  --train_dataset.data_mix "${DATA_MIX}" \
  --val_dataset.data_root_dir "${DATA_ROOT_DIR}" \
  --val_dataset.data_mix "${DATA_MIX}" \
  --wandb_project "${WANDB_PROJECT}" \
  ${EXTRA_ARGS}