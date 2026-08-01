#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""N7 评估速度误差统计。

Fast-BEV/N7 的 9 维框约定为 ``[x, y, z, l, w, h, yaw, vx, vy]``。
本模块只处理已经完成唯一 TP 匹配的框，避免速度指标重新定义匹配集合。
"""

import numpy as np


def _percentile_or_none(values, percentile):
    if len(values) == 0:
        return None
    return float(np.percentile(
        np.asarray(values, dtype=np.float32), percentile))


def _range_key(start, end):
    return '{}_{}m'.format(int(start), int(end))


def _empty_stats():
    return dict(
        velocity_tp_num=0,
        velocity_l2_mean=None,
        velocity_l2_p50=None,
        velocity_l2_p90=None,
        vx_abs_mean=None,
        vx_abs_p50=None,
        vx_abs_p90=None,
        vy_abs_mean=None,
        vy_abs_p50=None,
        vy_abs_p90=None,
        vx_mean=None,
        vy_mean=None)


def _summarize_errors(errors):
    stats = _empty_stats()
    if not errors:
        return stats

    errors = np.stack(errors, axis=0).astype(np.float32, copy=False)
    abs_errors = np.abs(errors)
    velocity_l2 = np.linalg.norm(errors, axis=1)
    stats.update(
        velocity_tp_num=int(errors.shape[0]),
        # nuScenes mAVE 的单类基础量：二维速度向量 L2 误差均值。
        velocity_l2_mean=float(np.mean(velocity_l2)),
        velocity_l2_p50=_percentile_or_none(velocity_l2, 50),
        velocity_l2_p90=_percentile_or_none(velocity_l2, 90),
        vx_abs_mean=float(np.mean(abs_errors[:, 0])),
        vx_abs_p50=_percentile_or_none(abs_errors[:, 0], 50),
        vx_abs_p90=_percentile_or_none(abs_errors[:, 0], 90),
        vy_abs_mean=float(np.mean(abs_errors[:, 1])),
        vy_abs_p50=_percentile_or_none(abs_errors[:, 1], 50),
        vy_abs_p90=_percentile_or_none(abs_errors[:, 1], 90),
        vx_mean=float(np.mean(errors[:, 0])),
        vy_mean=float(np.mean(errors[:, 1])))
    return stats


def summarize_velocity_errors(matches, range_bins):
    """统计匹配 TP 的总体和按 GT x 距离分桶速度误差。

    缺少第 8/9 维或包含非有限速度的匹配不会被伪装成零误差，而是从速度
    统计中排除；``velocity_tp_num`` 明确给出实际速度样本数。
    """
    errors = []
    per_range_errors = {
        _range_key(start, end): []
        for start, end in range_bins
    }

    for match in matches:
        pred_box = np.asarray(match.get('pred_box'), dtype=np.float32).reshape(-1)
        gt_box = np.asarray(match.get('gt_box'), dtype=np.float32).reshape(-1)
        if pred_box.size < 9 or gt_box.size < 9:
            continue
        pred_velocity = pred_box[7:9]
        gt_velocity = gt_box[7:9]
        if not (np.all(np.isfinite(pred_velocity)) and
                np.all(np.isfinite(gt_velocity))):
            continue

        error = pred_velocity - gt_velocity
        errors.append(error)
        gt_x = float(gt_box[0])
        for start, end in range_bins:
            if float(start) <= gt_x < float(end):
                per_range_errors[_range_key(start, end)].append(error)
                break

    return dict(
        overall=_summarize_errors(errors),
        ranges={
            key: _summarize_errors(values)
            for key, values in per_range_errors.items()
        })


__all__ = ['summarize_velocity_errors']
