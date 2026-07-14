#!/usr/bin/env python3
"""Evaluate epoch checkpoints and write compact per-epoch summaries.

典型用法：4 卡训练机持续产出 ``epoch_*.pth``，单卡评估机同步
``work_dir`` 后运行本脚本。脚本会扫描 checkpoint，调用 ``tools/test.py``
生成结果 pkl，然后复用同一份 pkl 计算 eval 指标和 score 分布。
"""

import argparse
import ast
import csv
import datetime as dt
import json
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


EPOCH_RE = re.compile(r'^epoch_(\d+)\.pth$')
METRIC_SCHEMA_VERSION = 2
SUMMARY_COLUMNS = [
    'epoch',
    'status',
    'best',
    'mAP',
    'BEV_mAP@0.5',
    'mATE@2m',
    'mAOE_deg@2m',
    'mASE@2m',
    'car_center_AP',
    'truck_center_AP',
    'car_AP@0.5m',
    'car_AP@1m',
    'car_AP@2m',
    'car_AP@4m',
    'truck_AP@0.5m',
    'truck_AP@1m',
    'truck_AP@2m',
    'truck_AP@4m',
    'car_ATE@2m',
    'car_AOE_deg@2m',
    'car_ASE@2m',
    'truck_ATE@2m',
    'truck_AOE_deg@2m',
    'truck_ASE@2m',
    'car_bev_AP@0.5',
    'truck_bev_AP@0.5',
    'car_bev_AP@0.25',
    'truck_bev_AP@0.25',
    'car_gt',
    'truck_gt',
    'car_det',
    'truck_det',
    'eval_det',
    'car_score_p50',
    'car_score_p90',
    'car_score_p99',
    'car_score_ge_0.2',
    'truck_score_p50',
    'truck_score_p90',
    'truck_score_p99',
    'truck_score_ge_0.2',
    'result_pkl',
    'checkpoint',
    'error',
]
OVERALL_MARKDOWN_COLUMNS = [
    'epoch',
    'status',
    'best',
    'mAP',
    'BEV_mAP@0.5',
    'mATE@2m',
    'mAOE_deg@2m',
    'mASE@2m',
    'eval_det',
]
CLASS_AP_MARKDOWN_COLUMNS = [
    'epoch',
    'best',
    'car_center_AP',
    'truck_center_AP',
    'car_AP@0.5m',
    'car_AP@1m',
    'car_AP@2m',
    'car_AP@4m',
    'truck_AP@0.5m',
    'truck_AP@1m',
    'truck_AP@2m',
    'truck_AP@4m',
]
CLASS_TP_ERROR_MARKDOWN_COLUMNS = [
    'epoch',
    'best',
    'car_ATE@2m',
    'car_AOE_deg@2m',
    'car_ASE@2m',
    'truck_ATE@2m',
    'truck_AOE_deg@2m',
    'truck_ASE@2m',
]
BEV_SCORE_MARKDOWN_COLUMNS = [
    'epoch',
    'best',
    'car_bev_AP@0.5',
    'truck_bev_AP@0.5',
    'car_bev_AP@0.25',
    'truck_bev_AP@0.25',
    'car_det',
    'truck_det',
    'car_score_p50',
    'car_score_p90',
    'car_score_p99',
    'car_score_ge_0.2',
    'truck_score_p50',
    'truck_score_p90',
    'truck_score_p99',
    'truck_score_ge_0.2',
]
RANGE_BINS = [
    ('0_20m', '0-20m'),
    ('20_40m', '20-40m'),
    ('40_60m', '40-60m'),
    ('60_80m', '60-80m'),
]
RANGE_ABS_MARKDOWN_COLUMNS = [
    'epoch',
    'best',
    'class',
    'range',
    'gt',
    'tp',
    'recall@2m',
    'x_mae',
    'x_p50',
    'x_p90',
    'y_mae',
    'y_p50',
    'y_p90',
    'z_mae',
    'z_p50',
    'z_p90',
]
RANGE_BIAS_MARKDOWN_COLUMNS = [
    'epoch',
    'best',
    'class',
    'range',
    'gt',
    'tp',
    'recall@2m',
    'x_mean',
    'y_mean',
    'z_mean',
]
RANGE_DETAIL_COLUMNS = (
    RANGE_ABS_MARKDOWN_COLUMNS +
    [key for key in RANGE_BIAS_MARKDOWN_COLUMNS
     if key not in RANGE_ABS_MARKDOWN_COLUMNS])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Scan epoch checkpoints, run single-GPU inference, '
        'evaluate metrics, and summarize score distributions.')
    parser.add_argument('config', help='Config file used for eval/test.')
    parser.add_argument('work_dir', help='Directory containing epoch_*.pth.')
    parser.add_argument(
        '--result-dir',
        default=None,
        help='Directory for result pkl, metric json, and summaries. '
        'Default: <work_dir>/test_results')
    parser.add_argument(
        '--epochs',
        type=int,
        nargs='+',
        help='Only process these epoch numbers.')
    parser.add_argument(
        '--epoch-range',
        type=int,
        nargs=2,
        metavar=('START', 'END'),
        help='Only process checkpoints in [START, END].')
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=['0.25', '0.5'],
        help='Metrics passed to dataset.evaluate, default: 0.25 0.5.')
    parser.add_argument(
        '--score-thresholds',
        type=float,
        nargs='+',
        default=[0.05, 0.1, 0.2],
        help='Score count thresholds for summary.')
    parser.add_argument(
        '--score-quantiles',
        type=float,
        nargs='+',
        default=[0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0],
        help='Quantiles used for score distribution.')
    parser.add_argument(
        '--best-key',
        default='mAP',
        help='Metric key used to mark the best checkpoint in summaries. '
        'Canonical mAP is center-distance mAP.')
    parser.add_argument(
        '--repo-root',
        default=None,
        help='Repository root. Default: parent of this tools directory.')
    parser.add_argument(
        '--python',
        default=sys.executable,
        help='Python executable used to launch tools/test.py.')
    parser.add_argument(
        '--test-script',
        default=None,
        help='Path to tools/test.py. Default: <repo_root>/tools/test.py.')
    parser.add_argument(
        '--cfg-options',
        nargs='*',
        help='Config overrides in key=value format, forwarded to tools/test.py '
        'and merged before building the eval dataset.')
    parser.add_argument(
        '--eval-options',
        nargs='*',
        help='Extra dataset.evaluate kwargs in key=value format.')
    parser.add_argument(
        '--cuda-visible-devices',
        default=None,
        help='Set CUDA_VISIBLE_DEVICES for tools/test.py.')
    parser.add_argument(
        '--rerun-inference',
        action='store_true',
        help='Regenerate result pkl even if it already exists.')
    parser.add_argument(
        '--rerun-eval',
        action='store_true',
        help='Recompute metrics json even if it already exists.')
    parser.add_argument(
        '--skip-inference',
        action='store_true',
        help='Only evaluate existing result pkl files; do not call tools/test.py.')
    parser.add_argument(
        '--watch',
        action='store_true',
        help='Keep polling work_dir for new stable epoch checkpoints.')
    parser.add_argument(
        '--poll-interval',
        type=int,
        default=300,
        help='Seconds between watch-mode scans.')
    parser.add_argument(
        '--stable-seconds',
        type=int,
        default=60,
        help='Skip checkpoints modified within this many seconds.')
    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='Stop after the first failed epoch.')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print planned actions without running inference/eval.')
    return parser.parse_args()


def to_builtin(value):
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, 'detach'):
        return to_builtin(value.detach().cpu().numpy())
    return value


def parse_key_value_options(raw_options):
    if not raw_options:
        return {}
    parsed = {}
    for item in raw_options:
        if '=' not in item:
            raise ValueError('Expected key=value option, got: {}'.format(item))
        key, raw_value = item.split('=', 1)
        try:
            value = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            value = raw_value
        parsed[key] = value
    return parsed


def parse_metric_values(raw_metrics):
    metrics = []
    for item in raw_metrics:
        try:
            metrics.append(float(item))
        except ValueError:
            metrics.append(item)
    return metrics


def metric_key(value):
    if isinstance(value, str):
        return value
    return '{:g}'.format(float(value))


def quantile_key(value):
    value = float(value)
    if value == 0.0:
        return 'min'
    if value == 1.0:
        return 'max'
    return 'p{:g}'.format(value * 100.0)


def threshold_key(value):
    return '{:g}'.format(float(value))


def repo_default():
    return Path(__file__).resolve().parents[1]


def resolve_paths(args):
    repo_root = Path(args.repo_root).resolve() if args.repo_root else repo_default()
    config = Path(args.config)
    if not config.is_absolute():
        config = (Path.cwd() / config).resolve()
    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = (Path.cwd() / work_dir).resolve()
    result_dir = Path(args.result_dir) if args.result_dir else work_dir / 'test_results'
    if not result_dir.is_absolute():
        result_dir = (Path.cwd() / result_dir).resolve()
    test_script = Path(args.test_script) if args.test_script else repo_root / 'tools' / 'test.py'
    if not test_script.is_absolute():
        test_script = (Path.cwd() / test_script).resolve()
    return repo_root, config, work_dir, result_dir, test_script


def find_checkpoints(work_dir):
    checkpoints = []
    for path in work_dir.glob('epoch_*.pth'):
        match = EPOCH_RE.match(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    return sorted(checkpoints, key=lambda item: item[0])


def filter_checkpoints(checkpoints, epochs=None, epoch_range=None):
    selected = checkpoints
    if epochs:
        wanted = set(epochs)
        selected = [(epoch, path) for epoch, path in selected if epoch in wanted]
    if epoch_range:
        start, end = epoch_range
        selected = [(epoch, path) for epoch, path in selected if start <= epoch <= end]
    return selected


def checkpoint_is_stable(path, stable_seconds):
    if stable_seconds <= 0:
        return True
    age = time.time() - path.stat().st_mtime
    return age >= stable_seconds


def load_pickle(path):
    try:
        import mmcv
        return mmcv.load(str(path))
    except ImportError:
        with open(path, 'rb') as f:
            return pickle.load(f)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(to_builtin(payload), f, ensure_ascii=False, indent=2)
        f.write('\n')


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_tail(path, max_lines=80):
    if not path.exists():
        return ''
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    return ''.join(lines[-max_lines:])


def run_inference(args, repo_root, config, checkpoint, result_pkl, test_log, test_script):
    cmd = [
        args.python,
        str(test_script),
        str(config),
        str(checkpoint),
        '--out',
        str(result_pkl),
    ]
    if args.cfg_options:
        cmd += ['--cfg-options'] + list(args.cfg_options)

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(args.cuda_visible_devices)

    print('Running inference: {}'.format(' '.join(cmd)))
    print('  log: {}'.format(test_log))
    if args.dry_run:
        return

    test_log.parent.mkdir(parents=True, exist_ok=True)
    with open(test_log, 'w', encoding='utf-8') as log_f:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env)

    if proc.returncode != 0:
        tail = read_tail(test_log)
        raise RuntimeError(
            'tools/test.py failed with exit code {}. Log tail:\n{}'.format(
                proc.returncode, tail))


def build_eval_dataset(config, cfg_options):
    repo_root = repo_default()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    mmdet3d_root = os.environ.get('MMDET3D')
    if mmdet3d_root and Path(mmdet3d_root).exists() and mmdet3d_root not in sys.path:
        sys.path.insert(0, mmdet3d_root)

    from mmcv import Config
    from mmdet3d.datasets import build_dataset

    cfg = Config.fromfile(str(config))
    if cfg_options:
        cfg.merge_from_dict(cfg_options)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    prepare_eval_dataset_cfg(cfg.data.test)
    dataset = build_dataset(cfg.data.test)
    return cfg, dataset


def prepare_eval_dataset_cfg(ds_cfg):
    if isinstance(ds_cfg, list):
        for item in ds_cfg:
            prepare_eval_dataset_cfg(item)
        return
    if not isinstance(ds_cfg, dict):
        return
    ds_cfg['test_mode'] = True
    # data.test 里可能带 dataloader 参数，build_dataset 不能接收这些字段。
    for key in ['samples_per_gpu', 'workers_per_gpu', 'shuffle']:
        ds_cfg.pop(key, None)
    if 'dataset' in ds_cfg:
        prepare_eval_dataset_cfg(ds_cfg['dataset'])
    if 'datasets' in ds_cfg:
        prepare_eval_dataset_cfg(ds_cfg['datasets'])


def clean_eval_kwargs(cfg, metrics, eval_options):
    eval_kwargs = cfg.get('evaluation', {}).copy()
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule',
            'by_epoch', 'dynamic_intervals'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(eval_options)
    eval_kwargs['metric'] = metrics
    return eval_kwargs


def tensor_like_to_numpy(value):
    if value is None:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(value, 'tensor'):
        value = value.tensor
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    elif hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value)


def fallback_scores_labels(result):
    if isinstance(result, dict) and 'pts_bbox' in result:
        result = result['pts_bbox']
    if not isinstance(result, dict):
        raise TypeError('prediction result must be a dict, got {}'.format(type(result)))
    scores = tensor_like_to_numpy(result.get('scores_3d')).astype(np.float32).reshape(-1)
    labels = tensor_like_to_numpy(result.get('labels_3d')).astype(np.int64).reshape(-1)
    valid_len = min(scores.shape[0], labels.shape[0])
    return scores[:valid_len], labels[:valid_len]


def parse_result_scores(dataset, result):
    if hasattr(dataset, '_parse_det_result'):
        _, scores, labels = dataset._parse_det_result(result)
        return scores.astype(np.float32).reshape(-1), labels.astype(np.int64).reshape(-1), '_parse_det_result'
    scores, labels = fallback_scores_labels(result)
    return scores, labels, 'raw_result'


def compute_score_distribution(dataset, outputs, score_thresholds, score_quantiles):
    class_names = list(getattr(dataset, 'CLASSES', []))
    per_class = {class_id: [] for class_id in range(len(class_names))}
    parser_name = None

    for result in outputs:
        scores, labels, parser_name = parse_result_scores(dataset, result)
        valid_len = min(scores.shape[0], labels.shape[0])
        scores = scores[:valid_len]
        labels = labels[:valid_len]
        for class_id in per_class:
            class_scores = scores[labels == class_id]
            if class_scores.size:
                per_class[class_id].append(class_scores.astype(np.float32))

    classes = {}
    total_det = 0
    q_keys = [quantile_key(q) for q in score_quantiles]
    t_keys = [threshold_key(t) for t in score_thresholds]
    for class_id, class_name in enumerate(class_names):
        if per_class[class_id]:
            scores = np.concatenate(per_class[class_id], axis=0)
        else:
            scores = np.zeros((0,), dtype=np.float32)
        total_det += int(scores.size)

        if scores.size:
            q_values = np.quantile(scores, score_quantiles)
            quantiles = {
                key: float(value)
                for key, value in zip(q_keys, q_values.tolist())
            }
        else:
            quantiles = {key: None for key in q_keys}

        classes[class_name] = {
            'label': class_id,
            'num': int(scores.size),
            'quantiles': quantiles,
            'threshold_counts': {
                key: int((scores >= float(threshold)).sum())
                for key, threshold in zip(t_keys, score_thresholds)
            },
        }

    return {
        'parser': parser_name or 'none',
        'total_det': total_det,
        'quantile_keys': q_keys,
        'threshold_keys': t_keys,
        'classes': classes,
    }


def evaluate_result_pkl(args, config, result_pkl, dataset_cache):
    if dataset_cache.get('dataset') is None:
        cfg_options = parse_key_value_options(args.cfg_options)
        cfg, dataset = build_eval_dataset(config, cfg_options)
        dataset_cache['cfg'] = cfg
        dataset_cache['dataset'] = dataset

    cfg = dataset_cache['cfg']
    dataset = dataset_cache['dataset']
    outputs = load_pickle(result_pkl)
    metrics = parse_metric_values(args.metrics)
    eval_options = parse_key_value_options(args.eval_options)
    eval_kwargs = clean_eval_kwargs(cfg, metrics, eval_options)
    ret_dict = dataset.evaluate(outputs, **eval_kwargs)
    score_distribution = compute_score_distribution(
        dataset, outputs, args.score_thresholds, args.score_quantiles)
    return to_builtin(ret_dict), to_builtin(score_distribution)


def epoch_paths(result_dir, epoch):
    stem = 'epoch_{}'.format(epoch)
    return {
        'result_pkl': result_dir / '{}_val_results.pkl'.format(stem),
        'metrics_json': result_dir / '{}_metrics.json'.format(stem),
        'test_log': result_dir / '{}_test.log'.format(stem),
    }


def process_epoch(args, repo_root, config, result_dir, test_script, dataset_cache,
                  epoch, checkpoint):
    paths = epoch_paths(result_dir, epoch)
    result_pkl = paths['result_pkl']
    metrics_json = paths['metrics_json']
    test_log = paths['test_log']

    if (metrics_json.exists() and not args.rerun_eval and
            not args.rerun_inference):
        existing_record = load_json(metrics_json)
        if existing_record.get('status') == 'ok':
            return existing_record
        print('Retry epoch {}: previous status is {}'.format(
            epoch, existing_record.get('status', 'unknown')))

    if not checkpoint_is_stable(checkpoint, args.stable_seconds):
        print('Skip epoch {}: checkpoint is not stable yet ({})'.format(
            epoch, checkpoint))
        return None

    record = {
        'epoch': epoch,
        'checkpoint': str(checkpoint),
        'result_pkl': str(result_pkl),
        'metrics_json': str(metrics_json),
        'test_log': str(test_log),
        'created_at': dt.datetime.now().isoformat(timespec='seconds'),
        'status': 'ok',
    }

    try:
        need_inference = args.rerun_inference or not result_pkl.exists()
        if need_inference:
            if args.skip_inference:
                print('Skip epoch {}: result pkl is missing ({})'.format(
                    epoch, result_pkl))
                return None
            run_inference(args, repo_root, config, checkpoint, result_pkl,
                          test_log, test_script)

        if args.dry_run:
            record['status'] = 'dry_run'
            return record

        print('Evaluating epoch {}: {}'.format(epoch, result_pkl))
        metrics, score_distribution = evaluate_result_pkl(
            args, config, result_pkl, dataset_cache)
        record['metrics'] = metrics
        record['score_distribution'] = score_distribution
        write_json(metrics_json, record)
        return record
    except Exception as exc:
        record['status'] = 'failed'
        record['error'] = str(exc)
        write_json(metrics_json, record)
        if args.stop_on_error:
            raise
        print('Epoch {} failed: {}'.format(epoch, exc))
        return record


def load_existing_records(result_dir):
    records = []
    for path in sorted(result_dir.glob('epoch_*_metrics.json')):
        try:
            records.append(load_json(path))
        except Exception as exc:
            print('Ignore invalid metrics json {}: {}'.format(path, exc))
    records.sort(key=lambda item: int(item.get('epoch', -1)))
    return records


def get_metric(record, key):
    metrics = record.get('metrics') or {}
    if key == 'mAP':
        # 历史 JSON 的 mAP 曾表示 BEV 指标；只要 center alias 存在就优先
        # 使用它。仅 metric schema v2 及以后允许回退到裸 mAP。
        value = metrics.get('mAP/center_dist')
        if value is None:
            try:
                schema_version = int(
                    metrics.get('eval/metric_schema_version', 0))
            except (TypeError, ValueError):
                schema_version = 0
            value = metrics.get('mAP') if schema_version >= 2 else None
    else:
        value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def get_first_metric(record, keys):
    for key in keys:
        value = get_metric(record, key)
        if value is not None:
            return value
    return None


def get_count_metric(record, key):
    metrics = record.get('metrics') or {}
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def get_score_stat(record, class_name, quantile=None, threshold=None):
    score_distribution = record.get('score_distribution') or {}
    classes = score_distribution.get('classes') or {}
    stats = classes.get(class_name) or {}
    if quantile is not None:
        return (stats.get('quantiles') or {}).get(quantile)
    if threshold is not None:
        return (stats.get('threshold_counts') or {}).get(threshold)
    return stats.get('num')


def find_best_epoch(records, best_key):
    best_epoch = None
    best_value = None
    for record in records:
        if record.get('status') != 'ok':
            continue
        value = get_metric(record, best_key)
        if not isinstance(value, (int, float)):
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_epoch = int(record.get('epoch'))
    return best_epoch, best_value


def flatten_record(record, best_epoch):
    epoch = int(record.get('epoch', -1))
    metrics = record.get('metrics') or {}
    row = {
        'epoch': epoch,
        'status': record.get('status', ''),
        'best': '*' if best_epoch is not None and epoch == best_epoch else '',
        'mAP': get_metric(record, 'mAP'),
        'BEV_mAP@0.5': get_metric(record, 'mAP/bev_iou@0.5'),
        'mATE@2m': get_metric(record, 'mATE@2m'),
        'mAOE_deg@2m': get_metric(record, 'mAOE_deg@2m'),
        'mASE@2m': get_metric(record, 'mASE@2m'),
        'car_center_AP': get_first_metric(
            record, ['car/center_AP', 'car/center_dist_mAP']),
        'truck_center_AP': get_first_metric(
            record, ['truck/center_AP', 'truck/center_dist_mAP']),
        'car_AP@0.5m': get_metric(record, 'car/dist_AP@0.5m'),
        'car_AP@1m': get_metric(record, 'car/dist_AP@1m'),
        'car_AP@2m': get_metric(record, 'car/dist_AP@2m'),
        'car_AP@4m': get_metric(record, 'car/dist_AP@4m'),
        'truck_AP@0.5m': get_metric(record, 'truck/dist_AP@0.5m'),
        'truck_AP@1m': get_metric(record, 'truck/dist_AP@1m'),
        'truck_AP@2m': get_metric(record, 'truck/dist_AP@2m'),
        'truck_AP@4m': get_metric(record, 'truck/dist_AP@4m'),
        'car_ATE@2m': get_metric(record, 'car/ATE@2m'),
        'car_AOE_deg@2m': get_metric(record, 'car/AOE_deg@2m'),
        'car_ASE@2m': get_metric(record, 'car/ASE@2m'),
        'truck_ATE@2m': get_metric(record, 'truck/ATE@2m'),
        'truck_AOE_deg@2m': get_metric(record, 'truck/AOE_deg@2m'),
        'truck_ASE@2m': get_metric(record, 'truck/ASE@2m'),
        'car_bev_AP@0.5': get_metric(record, 'car/bev_AP@0.5'),
        'truck_bev_AP@0.5': get_metric(record, 'truck/bev_AP@0.5'),
        'car_bev_AP@0.25': get_metric(record, 'car/bev_AP@0.25'),
        'truck_bev_AP@0.25': get_metric(record, 'truck/bev_AP@0.25'),
        'car_gt': get_count_metric(record, 'car/gt_num'),
        'truck_gt': get_count_metric(record, 'truck/gt_num'),
        'car_det': get_count_metric(record, 'car/det_num'),
        'truck_det': get_count_metric(record, 'truck/det_num'),
        'eval_det': get_count_metric(record, 'eval/det_num'),
        'car_score_p50': get_score_stat(record, 'car', quantile='p50'),
        'car_score_p90': get_score_stat(record, 'car', quantile='p90'),
        'car_score_p99': get_score_stat(record, 'car', quantile='p99'),
        'car_score_ge_0.2': get_score_stat(record, 'car', threshold='0.2'),
        'truck_score_p50': get_score_stat(record, 'truck', quantile='p50'),
        'truck_score_p90': get_score_stat(record, 'truck', quantile='p90'),
        'truck_score_p99': get_score_stat(record, 'truck', quantile='p99'),
        'truck_score_ge_0.2': get_score_stat(record, 'truck', threshold='0.2'),
        'result_pkl': Path(record.get('result_pkl', '')).name,
        'checkpoint': Path(record.get('checkpoint', '')).name,
        'error': record.get('error', ''),
    }

    # Keep fallback values if a future dataset changes metric names.
    if row['eval_det'] is None and 'eval/det_num' in metrics:
        row['eval_det'] = metrics['eval/det_num']
    return row


def flatten_range_detail_records(records, best_epoch):
    rows = []
    for record in records:
        epoch = int(record.get('epoch', -1))
        best = '*' if best_epoch is not None and epoch == best_epoch else ''
        for class_name in ['car', 'truck']:
            for range_key, range_label in RANGE_BINS:
                prefix = '{}/range_{}'.format(class_name, range_key)
                gt_num = get_count_metric(
                    record, '{}/gt_num'.format(prefix))
                tp_num = get_count_metric(
                    record, '{}/tp_num'.format(prefix))
                recall = get_metric(
                    record, '{}/recall@2m'.format(prefix))
                if recall is None and gt_num not in (None, 0) and tp_num is not None:
                    recall = float(tp_num) / float(gt_num)
                rows.append({
                    'epoch': epoch,
                    'best': best,
                    'class': class_name,
                    'range': range_label,
                    'gt': gt_num,
                    'tp': tp_num,
                    'recall@2m': recall,
                    'x_mae': get_metric(record, '{}/x_abs_mean'.format(prefix)),
                    'x_p50': get_metric(record, '{}/x_abs_p50'.format(prefix)),
                    'x_p90': get_metric(record, '{}/x_abs_p90'.format(prefix)),
                    'y_mae': get_metric(record, '{}/y_abs_mean'.format(prefix)),
                    'y_p50': get_metric(record, '{}/y_abs_p50'.format(prefix)),
                    'y_p90': get_metric(record, '{}/y_abs_p90'.format(prefix)),
                    'z_mae': get_metric(record, '{}/z_abs_mean'.format(prefix)),
                    'z_p50': get_metric(record, '{}/z_abs_p50'.format(prefix)),
                    'z_p90': get_metric(record, '{}/z_abs_p90'.format(prefix)),
                    'x_mean': get_metric(record, '{}/x_mean'.format(prefix)),
                    'y_mean': get_metric(record, '{}/y_mean'.format(prefix)),
                    'z_mean': get_metric(record, '{}/z_mean'.format(prefix)),
                })
    return rows


def format_cell(value):
    if value is None:
        return ''
    if isinstance(value, float):
        if np.isnan(value):
            return 'nan'
        return '{:.4f}'.format(value)
    return str(value)


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in SUMMARY_COLUMNS})


def write_range_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=RANGE_DETAIL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in RANGE_DETAIL_COLUMNS})


def write_markdown_table(f, columns, rows):
    f.write('| ' + ' | '.join(columns) + ' |\n')
    f.write('| ' + ' | '.join(['---'] * len(columns)) + ' |\n')
    for row in rows:
        cells = [format_cell(row.get(key, '')) for key in columns]
        f.write('| ' + ' | '.join(cells) + ' |\n')


def write_markdown(path, rows, range_rows, best_key, best_epoch, best_value):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# Epoch Eval Summary\n\n')
        f.write('- best_key: `{}`\n'.format(best_key))
        if best_epoch is not None:
            f.write('- best_epoch: `{}` (`{}` = `{:.6f}`)\n'.format(
                best_epoch, best_key, float(best_value)))
        else:
            f.write('- best_epoch: none\n')
        f.write('\n')

        f.write('## NuScenes-Like Overall\n\n')
        f.write('`mAP` is the canonical center-distance mAP: first average '
                'AP@0.5m/1m/2m/4m within each class, then average classes '
                'with GT. `BEV_mAP@0.5` is reported separately and is not '
                'an alias of `mAP`.\n\n')
        write_markdown_table(f, OVERALL_MARKDOWN_COLUMNS, rows)

        f.write('\n## Center-Distance AP By Class\n\n')
        write_markdown_table(f, CLASS_AP_MARKDOWN_COLUMNS, rows)

        f.write('\n## TP Errors By Class\n\n')
        f.write('ATE/AOE/ASE are computed on TP matches with center distance <= 2m.\n\n')
        write_markdown_table(f, CLASS_TP_ERROR_MARKDOWN_COLUMNS, rows)

        f.write('\n## BEV And Score Detail\n\n')
        write_markdown_table(f, BEV_SCORE_MARKDOWN_COLUMNS, rows)

        if range_rows:
            f.write('\n## TP XYZ Absolute Error By GT X Range\n\n')
            f.write(
                '`gt` is all filtered GT in the range; `tp` is the number of '
                'unique center-distance matches <=2m, and `recall@2m=tp/gt`. '
                'MAE is mean(abs(pred-gt)) over those matched TPs; MAE and '
                'absolute-error p50/p90 are in meters. Recall exposes misses, '
                'while MAE describes localization accuracy after matching.\n\n')
            write_markdown_table(f, RANGE_ABS_MARKDOWN_COLUMNS, range_rows)

            f.write('\n## TP XYZ Signed Bias By GT X Range\n\n')
            f.write(
                'Signed mean(pred-gt) is retained for systematic-bias '
                'diagnosis. Positive and negative errors may cancel, so this '
                'table must not be interpreted as absolute accuracy.\n\n')
            write_markdown_table(f, RANGE_BIAS_MARKDOWN_COLUMNS, range_rows)


def write_summaries(result_dir, best_key):
    records = load_existing_records(result_dir)
    best_epoch, best_value = find_best_epoch(records, best_key)
    rows = [flatten_record(record, best_epoch) for record in records]
    range_rows = flatten_range_detail_records(records, best_epoch)

    write_csv(result_dir / 'eval_summary.csv', rows)
    write_range_csv(result_dir / 'eval_range_summary.csv', range_rows)
    write_markdown(result_dir / 'eval_summary.md', rows, range_rows, best_key,
                   best_epoch, best_value)
    write_json(result_dir / 'eval_summary.json', {
        'metric_schema_version': METRIC_SCHEMA_VERSION,
        'metric_definitions': {
            'mAP': 'mean per-class center AP over 0.5m/1m/2m/4m',
            'BEV_mAP@0.5': 'mean per-class BEV IoU AP at 0.5',
            'range_recall@2m': 'unique center-distance TP<=2m / range GT',
            'range_xyz_mae': 'mean(abs(pred-gt)) over matched TP<=2m',
        },
        'best_key': best_key,
        'best_epoch': best_epoch,
        'best_value': best_value,
        'rows': rows,
        'range_rows': range_rows,
        'records': records,
    })
    print('Summary updated: {}'.format(result_dir / 'eval_summary.md'))


def process_once(args, repo_root, config, work_dir, result_dir, test_script,
                 dataset_cache):
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = filter_checkpoints(
        find_checkpoints(work_dir),
        epochs=args.epochs,
        epoch_range=args.epoch_range)

    if not checkpoints:
        print('No epoch checkpoints found in {}'.format(work_dir))
        write_summaries(result_dir, args.best_key)
        return

    for epoch, checkpoint in checkpoints:
        process_epoch(args, repo_root, config, result_dir, test_script,
                      dataset_cache, epoch, checkpoint)

    write_summaries(result_dir, args.best_key)


def main():
    args = parse_args()
    repo_root, config, work_dir, result_dir, test_script = resolve_paths(args)
    dataset_cache = {}

    if not config.exists():
        raise FileNotFoundError(config)
    if not work_dir.exists():
        raise FileNotFoundError(work_dir)
    if not test_script.exists():
        raise FileNotFoundError(test_script)

    print('config: {}'.format(config))
    print('work_dir: {}'.format(work_dir))
    print('result_dir: {}'.format(result_dir))
    print('test_script: {}'.format(test_script))

    while True:
        process_once(args, repo_root, config, work_dir, result_dir, test_script,
                     dataset_cache)
        if not args.watch:
            break
        print('Watch mode: sleep {} seconds'.format(args.poll_interval))
        time.sleep(args.poll_interval)


if __name__ == '__main__':
    main()
