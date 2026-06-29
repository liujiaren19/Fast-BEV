#!/bin/bash
set -e

TRAIN_PID=""

# 清理函数 - 只清理当前脚本启动的训练进程组
cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM

    if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
        echo "清理当前脚本启动的训练进程..."
        kill -TERM -- "-${TRAIN_PID}" 2>/dev/null || true
        sleep 5
        kill -KILL -- "-${TRAIN_PID}" 2>/dev/null || true
        wait "${TRAIN_PID}" 2>/dev/null || true
    fi

    exit "${exit_code}"
}

# 设置信号处理
trap cleanup EXIT INT TERM

export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3 # 使用第1、2块GPU（注意CUDA编号从0开始）
# export CUDA_VISIBLE_DEVICES=0,2,4,6
export PYTHONPATH=$PYTHONPATH:$(pwd)

CONFIG="${CONFIG:-configs/fastbev/custom/custom_fastbev_6v_r18_dist_train_ljr.py}"
WORK_DIR="${WORK_DIR:-work_dirs/n7_6v_704_256/20251017_20251030_20251031_20251203_gpu4_batch24_work_8_260626}"
TRAIN_BATCH="${TRAIN_BATCH:-1}"
EVAL_BATCH="${EVAL_BATCH:-${TRAIN_BATCH}}"
WORKERS="${WORKERS:-4}"

MASTER_PORT=$(comm -23 <(seq 29500 29600 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

echo "[INFO] CONFIG=${CONFIG}"
echo "[INFO] WORK_DIR=${WORK_DIR}"
echo "[INFO] TRAIN_BATCH=${TRAIN_BATCH}, EVAL_BATCH=${EVAL_BATCH}, WORKERS=${WORKERS}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# 使用适合PyTorch 1.10.0的启动方式
setsid python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    tools/train.py \
    "${CONFIG}" \
    --work-dir="${WORK_DIR}" \
    --launcher=pytorch \
    --cfg-options \
    data.samples_per_gpu="${TRAIN_BATCH}" \
    data.workers_per_gpu="${WORKERS}" \
    data.val.samples_per_gpu="${EVAL_BATCH}" \
    data.test.samples_per_gpu="${EVAL_BATCH}" &

TRAIN_PID=$!
wait "${TRAIN_PID}"
