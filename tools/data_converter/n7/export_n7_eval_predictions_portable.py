#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 MMDetection3D eval 预测 pkl 转成 Windows 可直接读取的 numpy-only pkl。

``tools/test.py --out`` 保存的 ``boxes_3d`` 通常是
``LiDARInstance3DBoxes``，直接在轻量 Windows 环境反序列化会尝试加载旧版
MMDetection3D/CUDA 扩展。本脚本应在能正常运行 eval 的服务器环境执行一次，
输出只包含 numpy 数组的 pkl，再把输出文件复制到 Windows 做离线可视化。

示例：

    python tools/data_converter/n7/export_n7_eval_predictions_portable.py \
      --input work_dirs/run/test_results/epoch_5_val_results.pkl \
      --output work_dirs/run/test_results/epoch_5_val_results_portable.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from tools.n7_box_origin import boxes_to_numpy as n7_boxes_to_numpy
except ModuleNotFoundError:
    TOOL_DIR = Path(__file__).resolve().parents[2]
    if str(TOOL_DIR) not in sys.path:
        sys.path.insert(0, str(TOOL_DIR))
    from n7_box_origin import boxes_to_numpy as n7_boxes_to_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把 eval 预测结果转换成不依赖 MMDetection3D 类的 numpy-only pkl。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True,
                        help="tools/test.py --out 生成的原始预测 pkl")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出路径；默认在输入文件名后增加 _portable")
    parser.add_argument("--overwrite", action="store_true",
                        help="允许覆盖已有输出")
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem + "_portable" + input_path.suffix)


def tensor_like_to_numpy(value, dtype=None) -> np.ndarray:
    if hasattr(value, "tensor"):
        value = value.tensor
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        array = value.numpy()
    else:
        array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def unwrap_prediction_result(result):
    if isinstance(result, dict) and "pts_bbox" in result:
        return result["pts_bbox"]
    return result


def boxes_to_center_numpy(boxes, box_origin: Optional[str]) -> np.ndarray:
    """通过 N7 公共 origin 契约输出重心 origin numpy box。"""
    return n7_boxes_to_numpy(
        boxes,
        target_origin="center",
        source_origin=box_origin,
        box_dim=None,
        dtype=np.float32)


def portable_result(result, index: int) -> Dict:
    result = unwrap_prediction_result(result)
    if not isinstance(result, dict):
        raise TypeError("第 {} 帧预测结果不是 dict：{}".format(index, type(result).__name__))

    boxes = result.get("boxes_3d", result.get("bboxes_3d"))
    scores = result.get("scores_3d", result.get("scores"))
    labels = result.get("labels_3d", result.get("labels"))
    if boxes is None or scores is None or labels is None:
        raise KeyError("第 {} 帧缺少 boxes/scores/labels".format(index))

    boxes_array = boxes_to_center_numpy(boxes, result.get("box_origin"))
    scores_array = tensor_like_to_numpy(scores, dtype=np.float32).reshape(-1)
    labels_array = tensor_like_to_numpy(labels, dtype=np.int64).reshape(-1)
    if not (len(boxes_array) == len(scores_array) == len(labels_array)):
        raise ValueError(
            "第 {} 帧数量不一致：boxes={} scores={} labels={}".format(
                index, len(boxes_array), len(scores_array), len(labels_array)))

    return {
        "boxes_3d": boxes_array,
        "scores_3d": scores_array,
        "labels_3d": labels_array,
        # 告诉可视化脚本输出 z 已经是重心，不能再补 h/2。
        "box_origin": "center",
    }


def load_results(path: Path) -> List:
    with path.open("rb") as file_obj:
        payload = pickle.load(file_obj)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "outputs", "predictions"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise TypeError("预测 pkl 最终必须包含 list，实际为：{}".format(type(payload).__name__))


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve() if args.output is not None
        else default_output_path(input_path))
    if not input_path.is_file():
        raise FileNotFoundError("找不到输入 pkl：{}".format(input_path))
    if output_path.exists() and not args.overwrite:
        raise FileExistsError("输出已存在；如需覆盖请加 --overwrite：{}".format(output_path))
    if output_path == input_path:
        raise ValueError("--output 不能和 --input 相同")

    results = load_results(input_path)
    portable = []
    detection_count = 0
    for index, result in enumerate(results):
        converted = portable_result(result, index)
        portable.append(converted)
        detection_count += len(converted["scores_3d"])
        if (index + 1) % 1000 == 0:
            print("[convert] {}/{} frames".format(index + 1, len(results)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with temp_path.open("wb") as file_obj:
            # protocol=4 兼容仍在使用 Python 3.8 的 Windows 环境。
            pickle.dump(portable, file_obj, protocol=4)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    print("[done] frames={} detections={} output={}".format(
        len(portable), detection_count, output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
