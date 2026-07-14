#!/usr/bin/env python3
"""N7 LiDAR 3D box origin 的轻量公共转换工具。

本模块只依赖 numpy，供 dataset eval、离线可视化、生产 JSON/pkl 导出和
板端 Python 后处理共同复用。N7/Fast-BEV 的 box 排列约定为
``[x, y, z, l, w, h, yaw, ...]``，其中：

- ``center``：``z`` 是 3D 框重心；
- ``bottom``：``z`` 是底面中心；
- ``native``：不解释也不修改输入 ``z``，只完成 numpy/shape 转换。

MMDetection3D 的 ``LiDARInstance3DBoxes.tensor`` 使用底中心，而
``gravity_center`` 使用重心。普通 numpy 数组没有可推断的 origin，因此调用方
若需要转换，必须通过 ``source_origin`` 明确声明；未声明时保持旧文件兼容，
不会猜测并重复补偿 ``h / 2``。
"""

from __future__ import annotations

from typing import Optional

import numpy as np


BOX_ORIGIN_CENTER = 'center'
BOX_ORIGIN_BOTTOM = 'bottom'
BOX_ORIGIN_NATIVE = 'native'
SUPPORTED_BOX_ORIGINS = (
    BOX_ORIGIN_CENTER,
    BOX_ORIGIN_BOTTOM,
    BOX_ORIGIN_NATIVE,
)


def normalize_box_origin(
    origin: Optional[str],
    *,
    allow_none: bool = True,
    allow_native: bool = True,
) -> Optional[str]:
    """校验并归一化 box origin 字符串。"""
    if origin is None:
        if allow_none:
            return None
        raise ValueError('box_origin must be explicitly set')
    normalized = str(origin).strip().lower()
    supported = {BOX_ORIGIN_CENTER, BOX_ORIGIN_BOTTOM}
    if allow_native:
        supported.add(BOX_ORIGIN_NATIVE)
    if normalized not in supported:
        raise ValueError(
            'unsupported box_origin {!r}; expected one of {}'.format(
                origin, sorted(supported)))
    return normalized


def tensor_like_to_numpy(value, dtype=np.float32) -> np.ndarray:
    """把 tensor/list/ndarray 转成 numpy，不隐式修改 box origin。"""
    if value is None:
        return np.asarray([], dtype=dtype)
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        array = value.numpy()
    else:
        array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def convert_box_origin(
    boxes: np.ndarray,
    source_origin: str,
    target_origin: str,
    *,
    copy: bool = True,
) -> np.ndarray:
    """在 ``center`` 和 ``bottom`` 之间转换 numpy box 的 z。"""
    source_origin = normalize_box_origin(
        source_origin, allow_none=False, allow_native=False)
    target_origin = normalize_box_origin(
        target_origin, allow_none=False, allow_native=False)
    array = np.asarray(boxes)
    if array.ndim != 2:
        raise ValueError('boxes must be a 2D array, got shape {}'.format(array.shape))
    if array.shape[1] < 6:
        raise ValueError(
            'box origin conversion needs [x,y,z,l,w,h], got shape {}'.format(
                array.shape))
    result = array.copy() if copy else array
    if source_origin == target_origin:
        return result
    if source_origin == BOX_ORIGIN_BOTTOM:
        result[:, 2] += result[:, 5] * 0.5
    else:
        result[:, 2] -= result[:, 5] * 0.5
    return result


def boxes_to_numpy(
    boxes,
    *,
    target_origin: str = BOX_ORIGIN_CENTER,
    source_origin: Optional[str] = None,
    box_dim: Optional[int] = None,
    dtype=np.float32,
) -> np.ndarray:
    """把 N7 LiDAR box 对象或数组统一成指定 origin 的二维 numpy 数组。

    Args:
        boxes: ``LiDARInstance3DBoxes``、tensor、ndarray 或 list。
        target_origin: ``center``、``bottom`` 或 ``native``。
        source_origin: 普通数组的显式 origin。MMDetection3D box 对象会从
            ``gravity_center``/``tensor`` 契约转换，不依赖该参数。
        box_dim: 可选的输出前 N 维；例如 dataset eval 使用 ``7``，生产 pkl
            传 ``None`` 可保留速度等扩展维。
        dtype: 输出 dtype。

    未声明 ``source_origin`` 的普通数组只在 ``target_origin=native`` 或无需
    推断时原样返回，这是对旧 prediction pkl 的兼容策略。新导出文件应始终写
    ``box_origin`` metadata。
    """
    target_origin = normalize_box_origin(
        target_origin, allow_none=False, allow_native=True)
    source_origin = normalize_box_origin(
        source_origin, allow_none=True, allow_native=False)
    if box_dim is not None and int(box_dim) <= 0:
        raise ValueError('box_dim must be positive, got {}'.format(box_dim))

    is_box_object = hasattr(boxes, 'tensor')
    tensor_value = boxes.tensor if is_box_object else boxes
    array = tensor_like_to_numpy(tensor_value, dtype=dtype)
    if array.size == 0:
        empty_dim = (
            int(box_dim) if box_dim is not None else
            int(array.shape[-1]) if array.ndim >= 2 and array.shape[-1] > 0 else
            7)
        return np.zeros((0, empty_dim), dtype=dtype)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim > 2:
        array = array.reshape(-1, array.shape[-1])
    if array.ndim != 2:
        raise ValueError('boxes must resolve to 2D, got shape {}'.format(array.shape))
    if box_dim is not None:
        box_dim = int(box_dim)
        if array.shape[1] < box_dim:
            raise ValueError(
                'boxes have {} dims, fewer than requested {}'.format(
                    array.shape[1], box_dim))
        array = array[:, :box_dim]
    array = array.copy()

    if target_origin == BOX_ORIGIN_NATIVE:
        return array
    if array.shape[1] < 6:
        raise ValueError(
            'origin-aware boxes need at least 6 dims, got shape {}'.format(
                array.shape))

    if is_box_object:
        if target_origin == BOX_ORIGIN_BOTTOM:
            # MMDetection3D LiDARInstance3DBoxes.tensor 本身就是底中心语义。
            return array
        gravity_center = getattr(boxes, 'gravity_center', None)
        if gravity_center is None:
            raise ValueError(
                'box object {} lacks gravity_center; cannot convert to center'.format(
                    boxes.__class__.__name__))
        centers = tensor_like_to_numpy(gravity_center, dtype=dtype).reshape(-1, 3)
        if centers.shape[0] != array.shape[0]:
            raise ValueError(
                'gravity_center count {} does not match boxes {}'.format(
                    centers.shape[0], array.shape[0]))
        array[:, :3] = centers
        return array

    if source_origin is not None and source_origin != target_origin:
        array = convert_box_origin(
            array, source_origin, target_origin, copy=False)
    return array


__all__ = [
    'BOX_ORIGIN_BOTTOM',
    'BOX_ORIGIN_CENTER',
    'BOX_ORIGIN_NATIVE',
    'SUPPORTED_BOX_ORIGINS',
    'boxes_to_numpy',
    'convert_box_origin',
    'normalize_box_origin',
    'tensor_like_to_numpy',
]
