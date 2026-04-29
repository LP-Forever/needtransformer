#!/bin/bash
# =============================================================================
# Cosmos-Policy Training Script for dataset_cup (run on amlt cloud)
# =============================================================================

# 确保 uv 在 PATH 中（amlt setup 阶段安装的，但 command 阶段 PATH 不会继承）
export PATH="$HOME/.local/bin:$PATH"

# --------------------------- 可配置参数 --------------------------------------

# 分布式训练
NNODES=1
NPROC_PER_NODE=8                  # 云端 A100 x8
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500

# 数据
DATA_MIX="cup_4hz"
STAGE="finetune"
MAX_ACTION_DIM=17
MAX_STATE_DIM=17
DATASET_LEN=5000_0000
PARENT_DIR="/mnt/wangxiaofa/robot_dataset/lerobot-format"

# 训练
BATCH_SIZE=4
ACC_STEP=1                        # 8 GPU x 4 batch x 1 acc = 32 effective
MAX_ITER=30000
SAVE_ITER=5000
LOGGING_ITER=5

# 学习率与调度
LR=1e-4
SCHEDULER_CYCLE_LENGTHS=30000
SCHEDULER_WARM_UP_STEPS=1000

# 预训练模型路径（amlt 提交时放在 $AMLT_CODE_DIR/cosmos_pretrained/ 下）
CODE_DIR=${AMLT_CODE_DIR:-$PWD}
COSMOS_PRETRAINED="${CODE_DIR}/cosmos_pretrained/model-480p-16fps.pt"
T5_EMBEDDINGS="${CODE_DIR}/cosmos_pretrained/t5_embeddings_cup_4hz.pkl"

# WandB
WANDB_MODE="online"
WANDB_PROJECT="cosmos-CUP"
WANDB_ENTITY=""

# 输出
OUTPUT_ROOT="/mnt/wangxiaofa/cosmos-output"
TIMESTAMP=$(date +%Y%m%d_%H%M)
JOB_NAME="cosmos_2b_480p_cup_${TIMESTAMP}"

# --------------------------- 解析命令行参数 ----------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnodes)           NNODES="$2";           shift 2 ;;
        --nproc_per_node)   NPROC_PER_NODE="$2";   shift 2 ;;
        --node_rank)        NODE_RANK="$2";         shift 2 ;;
        --master_addr)      MASTER_ADDR="$2";       shift 2 ;;
        --master_port)      MASTER_PORT="$2";       shift 2 ;;
        --batch_size)       BATCH_SIZE="$2";        shift 2 ;;
        --data_mix)         DATA_MIX="$2";          shift 2 ;;
        --stage)            STAGE="$2";             shift 2 ;;
        --max_action_dim)   MAX_ACTION_DIM="$2";    shift 2 ;;
        --max_state_dim)    MAX_STATE_DIM="$2";     shift 2 ;;
        --acc_step)         ACC_STEP="$2";          shift 2 ;;
        --max_iter)         MAX_ITER="$2";          shift 2 ;;
        --save_iter)        SAVE_ITER="$2";         shift 2 ;;
        --logging_iter)     LOGGING_ITER="$2";      shift 2 ;;
        --lr)               LR="$2";                shift 2 ;;
        --output_root)      OUTPUT_ROOT="$2";       shift 2 ;;
        --job_name)         JOB_NAME="$2";          shift 2 ;;
        --wandb_mode)       WANDB_MODE="$2";        shift 2 ;;
        --wandb_project)    WANDB_PROJECT="$2";     shift 2 ;;
        --wandb_entity)     WANDB_ENTITY="$2";      shift 2 ;;
        --pretrained)       COSMOS_PRETRAINED="$2"; shift 2 ;;
        --t5_embeddings)    T5_EMBEDDINGS="$2";     shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# 设置输出目录
export IMAGINAIRE_OUTPUT_ROOT="${OUTPUT_ROOT}"

# WandB 环境变量
export WANDB_MODE="${WANDB_MODE}"
export WANDB_PROJECT="${WANDB_PROJECT}"
if [ -n "${WANDB_ENTITY}" ]; then
    export WANDB_ENTITY="${WANDB_ENTITY}"
fi

# 复制 t5_embeddings pkl 到代码期望的路径（代码硬编码从 parent_dir 下读取）
T5_PKL_SRC="${CODE_DIR}/cosmos_pretrained/t5_embeddings_${DATA_MIX}.pkl"
T5_PKL_DST="${PARENT_DIR}/t5_embeddings_${DATA_MIX}.pkl"
if [ -f "${T5_PKL_SRC}" ] && [ ! -f "${T5_PKL_DST}" ]; then
    echo "Copying t5_embeddings: ${T5_PKL_SRC} -> ${T5_PKL_DST}"
    cp "${T5_PKL_SRC}" "${T5_PKL_DST}"
else
    echo "t5_embeddings already at destination or source not found, skipping copy"
    echo "  src=${T5_PKL_SRC}  dst=${T5_PKL_DST}"
fi

# HuggingFace 离线模式：YAML 中已设置 HF_HUB_OFFLINE=0，这里不再覆盖
# 如需完全离线，取消下面两行的注释
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

echo "============================================"
echo "  Cosmos-Policy Training for dataset_cup"
echo "============================================"
echo "  Data mix:         ${DATA_MIX}"
echo "  Stage:            ${STAGE}"
echo "  GPUs:             ${NPROC_PER_NODE} x ${NNODES} nodes"
echo "  Batch size:       ${BATCH_SIZE} per GPU"
echo "  Grad accumulate:  ${ACC_STEP}"
echo "  Effective batch:  $((BATCH_SIZE * ACC_STEP * NPROC_PER_NODE))"
echo "  Max iterations:   ${MAX_ITER}"
echo "  Save every:       ${SAVE_ITER} steps"
echo "  Learning rate:    ${LR}"
echo "  Action dim:       ${MAX_ACTION_DIM}"
echo "  State dim:        ${MAX_STATE_DIM}"
echo "  Pretrained:       ${COSMOS_PRETRAINED}"
echo "  T5 embeddings:    ${T5_EMBEDDINGS}"
echo "  Output dir:       ${OUTPUT_ROOT}/cosmos_v2_finetune/cosmos_v2_finetune/${JOB_NAME}"
echo "  WandB:            ${WANDB_MODE} (project=${WANDB_PROJECT})"
echo "============================================"

cd "${CODE_DIR}/cosmos-policy"

uv run --no-sync --extra cu128 --group libero --python 3.10 \
  torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
  -m cosmos_policy.scripts.train \
    --config=cosmos_policy/config/config.py \
    --job_name=${JOB_NAME} \
    -- \
    experiment="cosmos_predict2_2b_480p_libero" \
    job.project="${WANDB_PROJECT}" \
    job.wandb_mode="${WANDB_MODE}" \
    trainer.max_iter=${MAX_ITER} \
    trainer.logging_iter=${LOGGING_ITER} \
    trainer.grad_accum_iter=${ACC_STEP} \
    checkpoint.save_iter=${SAVE_ITER} \
    checkpoint.load_path="${COSMOS_PRETRAINED}" \
    optimizer.lr=${LR} \
    scheduler.cycle_lengths="[${SCHEDULER_CYCLE_LENGTHS},100000000000000]" \
    scheduler.warm_up_steps="[${SCHEDULER_WARM_UP_STEPS},0]" \
    dataloader_train.batch_size=${BATCH_SIZE} \
    dataloader_train.dataset.data_mix=${DATA_MIX} \
    dataloader_train.dataset.stage=${STAGE} \
    dataloader_train.dataset.max_action_dim=${MAX_ACTION_DIM} \
    dataloader_train.dataset.max_state_dim=${MAX_STATE_DIM} \
    dataloader_train.dataset.dataset_len_one_epoch=${DATASET_LEN} \
    dataloader_train.dataset.parent_dir=${PARENT_DIR} \
    dataloader_train.dataset.t5_text_embeddings_path="${T5_EMBEDDINGS}"
