#!/bin/bash
# =============================================================================
# MemoryVLA Training Script (run on amlt cloud)
# =============================================================================
# Adapted from MemoryVLA/script/train/real_world/train_real.sh
# =============================================================================

# --------------------------- 可配置参数 --------------------------------------

# 分布式训练
NNODES=1
NPROC_PER_NODE=8                  # 云端 A100 x8
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500

# 数据
DATA_MIX="custom_finetuning"      # 改成你的数据集名
DATA_ROOT_DIR="/mnt/wangxiaofa/robot_dataset/memvla-rlds"  # RLDS 格式数据根目录

# 预训练模型
CODE_DIR=${AMLT_CODE_DIR:-$PWD}
PRETRAINED_CKPT="${CODE_DIR}/MemoryVLA/pretrained/CogACT-Large/checkpoints/CogACT-Large.pt"
# 也可用 OpenVLA backbone (适合 LIBERO 类数据):
# PRETRAINED_CKPT="${CODE_DIR}/MemoryVLA/pretrained/openvla-7b-prismatic/checkpoints/openvla-7b-prismatic.pt"
HF_TOKEN=""                        # 如果模型需要 HF 认证，填你的 token

# 训练
N_GPU=8
BS=32                               # per_device_batch_size
SHUFFLE_BUFFER_SIZE=32000           # 数据量小时可减小
SAVE_INTERVAL=5000
DP_STEP=4                           # repeated_diffusion_steps (diffusion 采样步数)
FUTURE_ACTION_WINDOW_SIZE=15        # action chunking 窗口大小
MAX_STEPS=20000                     # 总训练步数
LR=2e-5                             # 学习率

# Memory 模块参数
MEM_LENGTH=256                      # 记忆长度 (real_world 默认 256, 模拟默认 16)
DATALOADER_TYPE="stream"            # 实机数据推荐 stream, 模拟推荐 group
GROUP_SIZE=16                       # group dataloader 时的 group 大小

# WandB
WANDB_PROJECT="memvla"
WANDB_ENTITY=""                     # 你的 wandb entity

# 输出
RUN_ROOT_DIR="/mnt/wangxiaofa/memvla-output"
RUN_ID="memvla_custom"
TIMESTAMP=$(date +%Y%m%d_%H%M)
JOB_NAME="memvla_custom_${TIMESTAMP}"

# Resume
IS_RESUME=False
RESUME_STEP=0
RESUME_EPOCH=0

# --------------------------- 解析命令行参数 ----------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnodes)           NNODES="$2";           shift 2 ;;
        --nproc_per_node)   NPROC_PER_NODE="$2";   shift 2 ;;
        --node_rank)        NODE_RANK="$2";         shift 2 ;;
        --master_addr)      MASTER_ADDR="$2";       shift 2 ;;
        --master_port)      MASTER_PORT="$2";       shift 2 ;;
        --data_mix)         DATA_MIX="$2";          shift 2 ;;
        --data_root_dir)    DATA_ROOT_DIR="$2";     shift 2 ;;
        --pretrained_ckpt)  PRETRAINED_CKPT="$2";   shift 2 ;;
        --hf_token)         HF_TOKEN="$2";          shift 2 ;;
        --batch_size)       BS="$2";                shift 2 ;;
        --shuffle_buffer)   SHUFFLE_BUFFER_SIZE="$2"; shift 2 ;;
        --save_interval)    SAVE_INTERVAL="$2";     shift 2 ;;
        --dp_step)          DP_STEP="$2";           shift 2 ;;
        --future_window)    FUTURE_ACTION_WINDOW_SIZE="$2"; shift 2 ;;
        --max_steps)        MAX_STEPS="$2";         shift 2 ;;
        --lr)               LR="$2";                shift 2 ;;
        --mem_length)       MEM_LENGTH="$2";        shift 2 ;;
        --dataloader_type)  DATALOADER_TYPE="$2";   shift 2 ;;
        --run_root_dir)     RUN_ROOT_DIR="$2";      shift 2 ;;
        --run_id)           RUN_ID="$2";            shift 2 ;;
        --wandb_project)    WANDB_PROJECT="$2";     shift 2 ;;
        --wandb_entity)     WANDB_ENTITY="$2";      shift 2 ;;
        --is_resume)        IS_RESUME="$2";         shift 2 ;;
        --resume_step)      RESUME_STEP="$2";       shift 2 ;;
        --resume_epoch)     RESUME_EPOCH="$2";       shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# --------------------------- 打印配置 --------------------------------------

echo "============================================"
echo "  MemoryVLA Training"
echo "============================================"
echo "  Data mix:         ${DATA_MIX}"
echo "  Data root:        ${DATA_ROOT_DIR}"
echo "  Pretrained:       ${PRETRAINED_CKPT}"
echo "  GPUs:             ${NPROC_PER_NODE} x ${NNODES} nodes"
echo "  Batch size:       ${BS} per GPU"
echo "  Effective batch:  $((BS * NPROC_PER_NODE))"
echo "  Max steps:        ${MAX_STEPS}"
echo "  Save every:       ${SAVE_INTERVAL} steps"
echo "  Learning rate:    ${LR}"
echo "  Diffusion steps:  ${DP_STEP}"
echo "  Action window:    ${FUTURE_ACTION_WINDOW_SIZE}"
echo "  Memory length:    ${MEM_LENGTH}"
echo "  Dataloader type:  ${DATALOADER_TYPE}"
echo "  Shuffle buffer:   ${SHUFFLE_BUFFER_SIZE}"
echo "  Output dir:       ${RUN_ROOT_DIR}/${RUN_ID}"
echo "  WandB:            ${WANDB_PROJECT}"
echo "============================================"

# --------------------------- 数据准备 --------------------------------------
# 将 RLDS 数据放到 data_root_dir/custom_finetuning/1.0.0/ 下
# 如果数据在别的位置，做软链接
CUSTOM_DATA_DIR="${DATA_ROOT_DIR}/${DATA_MIX}/1.0.0"
if [ ! -d "${CUSTOM_DATA_DIR}" ]; then
    echo "WARNING: Data directory not found at ${CUSTOM_DATA_DIR}"
    echo "Make sure your RLDS-format data is placed under:"
    echo "  <data_root_dir>/<data_mix>/1.0.0/"
fi

# --------------------------- 启动训练 --------------------------------------

cd "${CODE_DIR}/MemoryVLA"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun \
  --nnodes=${NNODES} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
train.py \
  --pretrained_checkpoint "${PRETRAINED_CKPT}" \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix ${DATA_MIX} \
  --vla.expected_world_size ${NPROC_PER_NODE} \
  --vla.per_device_batch_size ${BS} \
  --vla.global_batch_size $((NPROC_PER_NODE * BS)) \
  --vla.learning_rate ${LR} \
  --vla.max_steps ${MAX_STEPS} \
  --data_root_dir ${DATA_ROOT_DIR} \
  --run_root_dir ${RUN_ROOT_DIR} \
  --run_id ${RUN_ID} \
  --image_aug True \
  --save_interval ${SAVE_INTERVAL} \
  --repeated_diffusion_steps ${DP_STEP} \
  --future_action_window_size ${FUTURE_ACTION_WINDOW_SIZE} \
  --action_model_type 'DiT-L' \
  --dataloader_type ${DATALOADER_TYPE} \
  --is_resume ${IS_RESUME} \
  --mem_length ${MEM_LENGTH} \
  --resume_step ${RESUME_STEP} \
  --resume_epoch ${RESUME_EPOCH} \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_entity "${WANDB_ENTITY}" \
  --hf_token "${HF_TOKEN}" \
  --vla.shuffle_buffer_size ${SHUFFLE_BUFFER_SIZE}