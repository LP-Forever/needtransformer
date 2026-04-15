#!/bin/bash
# =============================================================================
# Cosmos-Policy Training Script for dataset_cup (cup_4hz)
# =============================================================================

# --------------------------- 可配置参数 --------------------------------------

# 分布式训练
NNODES=1                          # 节点数
NPROC_PER_NODE=2                  # 每节点 GPU 数
NODE_RANK=0                       # 当前节点 rank（多节点时修改）
MASTER_ADDR="127.0.0.1"          # 多节点主节点地址
MASTER_PORT=29500                 # 多节点通信端口

# 数据
DATA_MIX="cup_4hz"                # 数据集 mixture 名（对应 mixtures.py 中的定义）
STAGE="finetune"                  # 训练阶段: "finetune" 或 "pretrain"
MAX_ACTION_DIM=17                 # action 维度（cup 数据集为 17 维）
MAX_STATE_DIM=17                  # state 维度（cup 数据集为 17 维）
DATASET_LEN=5000_0000             # pretrain 模式下每个 epoch 的采样量（finetune 模式忽略此参数）
PARENT_DIR="/home/v-wenhuitan/franka-action/cosmos-policy/dataset_cup"


# 训练
BATCH_SIZE=4                      # 每 GPU 的 batch size
ACC_STEP=2                        # 梯度累积步数（等效 batch = BATCH_SIZE * ACC_STEP * NPROC_PER_NODE）
MAX_ITER=30000                    # 最大训练步数
SAVE_ITER=5000                    # 每多少步保存 checkpoint
LOGGING_ITER=5                    # 每多少步打印一次 loss

# 学习率与调度
LR=1e-4                           # 学习率
SCHEDULER_CYCLE_LENGTHS=30000     # 第一个 cycle 的步数
SCHEDULER_WARM_UP_STEPS=1000      # warmup 步数

# WandB
WANDB_MODE="online"               # "online"=上传云端, "offline"=本地记录, "disabled"=关闭
WANDB_PROJECT="cosmos-CUP"        # WandB 项目名
WANDB_ENTITY=""                   # WandB 用户名/团队名（留空则用账号默认值）

# 输出
OUTPUT_ROOT="/data_16T/imaginaire4-output"  # 训练结果输出根目录（可改为你的路径）
JOB_NAME="cosmos_2b_480p_cup_20260414_1051"  # 作业名，结果保存在 ${OUTPUT_ROOT}/cosmos_v2_finetune/cosmos_v2_finetune/${JOB_NAME}

# --------------------------- 以下一般不需要修改 -------------------------------

# 解析命令行参数（可覆盖上方变量）
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
        --dataset_len)      DATASET_LEN="$2";       shift 2 ;;
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
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# 设置输出目录
export IMAGINAIRE_OUTPUT_ROOT="${OUTPUT_ROOT}"

# WandB 环境变量
export WANDB_API_KEY="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"
export WANDB_MODE="${WANDB_MODE}"
export WANDB_PROJECT="${WANDB_PROJECT}"
if [ -n "${WANDB_ENTITY}" ]; then
    export WANDB_ENTITY="${WANDB_ENTITY}"
fi

# conda 环境（按需修改）
# export PATH=/home/aiscuser/.conda/envs/lerobot/bin:$PATH
# export LD_LIBRARY_PATH=/home/aiscuser/.conda/envs/lerobot/lib:$LD_LIBRARY_PATH

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
echo "  Output dir:       ${OUTPUT_ROOT}/cosmos_v2_finetune/cosmos_v2_finetune/${JOB_NAME}"
echo "  WandB:            ${WANDB_MODE} (project=${WANDB_PROJECT})"
echo "============================================"

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
    optimizer.lr=${LR} \
    scheduler.cycle_lengths="[${SCHEDULER_CYCLE_LENGTHS},100000000000000]" \
    scheduler.warm_up_steps="[${SCHEDULER_WARM_UP_STEPS},0]" \
    dataloader_train.batch_size=${BATCH_SIZE} \
    dataloader_train.dataset.data_mix=${DATA_MIX} \
    dataloader_train.dataset.stage=${STAGE} \
    dataloader_train.dataset.max_action_dim=${MAX_ACTION_DIM} \
    dataloader_train.dataset.max_state_dim=${MAX_STATE_DIM} \
    dataloader_train.dataset.dataset_len_one_epoch=${DATASET_LEN} \
    dataloader_train.dataset.parent_dir=${PARENT_DIR}


