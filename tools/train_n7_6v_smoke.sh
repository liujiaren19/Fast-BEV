#!/usr/bin/env bash
set -e

# N7 自采 6V Fast-BEV 训练 smoke。
# 这个脚本只做一件事：用当前 one-clip pkl 跑通 train.py。
# 如需改 batch、workers、epoch、pkl 路径，直接改下面命令里的对应字段。

cd /root/autodl-tmp/Fast-BEV

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/autodl-tmp/Fast-BEV:${PYTHONPATH}
export PYTHONUNBUFFERED=1

python -u tools/train.py configs/fastbev/custom/custom_fastbev_6v_r18.py \
  --work-dir work_dirs/debug_n7_6v_oneclip \
  --seed 0 \
  --no-validate \
  --cfg-options \
  data.train.ann_file=data/nuscenes/pkl/od_2k/custom_fastbev_20251031_164821_1_infos_test_20260623.pkl \
  data.val.ann_file=data/nuscenes/pkl/od_2k/custom_fastbev_20251031_164821_1_infos_test_20260623.pkl \
  data.test.ann_file=data/nuscenes/pkl/od_2k/custom_fastbev_20251031_164821_1_infos_test_20260623.pkl \
  total_epochs=1 \
  data.samples_per_gpu=2 \
  data.workers_per_gpu=4 \
  log_config.interval=10 \
  checkpoint_config.interval=999
