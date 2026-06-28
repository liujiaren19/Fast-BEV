#!/usr/bin/env bash
set -e

# N7 自采 6V Fast-BEV test/eval 脚本。
# 用途：
#   1. 读取指定 checkpoint，在 val/test pkl 上执行推理；
#   2. 保存 tools/test.py 输出的预测结果 pkl；
#   3. 通过 CustomMultiViewDataset.evaluate 输出 BEV IoU AP / 中心距离 AP 等指标。
#
# 默认路径按当前 22W 全量训练实验填写；如需临时切换 checkpoint、batch 或 pkl，
# 可在命令前通过环境变量覆盖，例如：
#
#   CKPT=work_dirs/.../epoch_2.pth TEST_BATCH=16 bash tools/test_n7_6v_eval.sh
#
# 注意：
#   - 当前脚本只做指标与预测 pkl 输出，不使用 tools/test.py --show；
#   - N7 多相机效果图建议后续用 N7 专用可视化脚本读取这里保存的 OUT_PKL。
#   - 默认关闭 fp16，让 test/eval 更接近后续 ONNX/板端 FP32 导出链路；
#     如需对比 fp16 推理速度或指标，可设置 DISABLE_FP16=0。

# 自动切到工程根目录，避免依赖固定的 /root/autodl-tmp/Fast-BEV 路径。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

# 优先使用内网全量训练配置；如果该文件不存在，则回退到已提交的 N7 704x256 配置。
CONFIG="${CONFIG:-configs/fastbev/custom/custom_fastbev_6v_r18_dist_train_ljr.py}"
if [ ! -f "${CONFIG}" ]; then
  CONFIG="configs/fastbev/custom/custom_fastbev_6v_r18_n7_704x256.py"
fi

VAL_PKL="${VAL_PKL:-./data/N7_704_256/pkl/custom_fastbev_20251017-20251030-20251031-20251203_infos_val_20260625.pkl}"
WEIGHT_DIR="${WEIGHT_DIR:-work_dirs/n7_6v_704_256/20251017_20251030_20251031_20251203_gpu4_batch24_work_8_260626}"

# 如果没有显式指定 CKPT，优先使用 latest.pth；否则自动选择编号最大的 epoch_*.pth。
if [ -z "${CKPT:-}" ]; then
  if [ -f "${WEIGHT_DIR}/latest.pth" ]; then
    CKPT="${WEIGHT_DIR}/latest.pth"
  else
    CKPT="$(ls -1v "${WEIGHT_DIR}"/epoch_*.pth 2>/dev/null | tail -n 1)"
  fi
fi

if [ -z "${CKPT}" ] || [ ! -f "${CKPT}" ]; then
  echo "[ERROR] 找不到 checkpoint。请设置 CKPT=/path/to/epoch_x.pth 或检查 WEIGHT_DIR=${WEIGHT_DIR}" >&2
  exit 1
fi

if [ ! -f "${VAL_PKL}" ]; then
  echo "[ERROR] 找不到 val/test pkl：${VAL_PKL}" >&2
  exit 1
fi

TEST_BATCH="${TEST_BATCH:-8}"
WORKERS="${WORKERS:-8}"
DISABLE_FP16="${DISABLE_FP16:-1}"
PROFILE_TEST="${PROFILE_TEST:-1}"
PROFILE_INTERVAL="${PROFILE_INTERVAL:-10}"
OUT_DIR="${OUT_DIR:-${WEIGHT_DIR}/test_results}"
OUT_PKL="${OUT_PKL:-${OUT_DIR}/$(basename "${CKPT}" .pth)_val_results.pkl}"
mkdir -p "${OUT_DIR}"

echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] CONFIG=${CONFIG}"
echo "[INFO] CKPT=${CKPT}"
echo "[INFO] VAL_PKL=${VAL_PKL}"
echo "[INFO] OUT_PKL=${OUT_PKL}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] TEST_BATCH=${TEST_BATCH}, WORKERS=${WORKERS}"
echo "[INFO] DISABLE_FP16=${DISABLE_FP16}"
echo "[INFO] PROFILE_TEST=${PROFILE_TEST}, PROFILE_INTERVAL=${PROFILE_INTERVAL}"

# tools/test.py 会在 cfg.fp16 存在时自动 wrap_fp16_model。
# 这里默认写 fp16=None，避免 test/eval 与后续 ONNX/FP32 导出链路不一致。
CFG_OPTIONS=(
  data.test.ann_file="${VAL_PKL}"
  data.val.ann_file="${VAL_PKL}"
  data.test.samples_per_gpu="${TEST_BATCH}"
  data.workers_per_gpu="${WORKERS}"
)

if [ "${DISABLE_FP16}" = "1" ]; then
  CFG_OPTIONS+=(fp16=None)
fi

PROFILE_ARGS=()
if [ "${PROFILE_TEST}" = "1" ]; then
  PROFILE_ARGS+=(--profile-test --profile-interval "${PROFILE_INTERVAL}")
fi

python -u tools/test.py "${CONFIG}" "${CKPT}" \
  --out "${OUT_PKL}" \
  --eval 0.25 0.5 \
  "${PROFILE_ARGS[@]}" \
  --cfg-options "${CFG_OPTIONS[@]}"
