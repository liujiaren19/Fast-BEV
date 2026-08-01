#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 legacy MMDetection 环境运行有限 iteration 的 N7 训练吞吐门禁。

该入口使用项目真实 dataset/model/runner/optimizer/fp16 hook，运行指定 warmup
和测量 iteration 后自然退出；输出 data_time、iter_time、samples/s 和 CUDA
峰值显存 JSON。可先用 ``--warmup-iters 0 --measure-iters 1`` 做单 batch
forward/backward smoke，再用默认 10+50 iteration 做 S0/GEOM1-A 对比。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, level):
    import numpy as np
    return float(np.quantile(np.asarray(values, dtype=np.float64), level))


def register_benchmark_hook():
    import torch
    from mmcv.runner import HOOKS, Hook

    @HOOKS.register_module(force=True)
    class N7TrainingBenchmarkHook(Hook):
        def __init__(self, output_path, warmup_iters, measure_iters,
                     samples_per_gpu, config_path, config_sha256):
            self.output_path = Path(output_path)
            self.warmup_iters = int(warmup_iters)
            self.measure_iters = int(measure_iters)
            self.samples_per_gpu = int(samples_per_gpu)
            self.config_path = str(config_path)
            self.config_sha256 = str(config_sha256)
            self.data_times = []
            self.iter_times = []
            self.last_end = None
            self.iter_start = None

        def before_run(self, runner):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.last_end = time.perf_counter()

        def before_train_iter(self, runner):
            torch.cuda.synchronize()
            now = time.perf_counter()
            data_time = now - self.last_end
            if runner.iter >= self.warmup_iters:
                self.data_times.append(data_time)
            if runner.iter == self.warmup_iters:
                torch.cuda.reset_peak_memory_stats()
            self.iter_start = now

        def after_train_iter(self, runner):
            torch.cuda.synchronize()
            now = time.perf_counter()
            if runner.iter >= self.warmup_iters:
                self.iter_times.append(now - self.iter_start)
            self.last_end = now

        def after_run(self, runner):
            if getattr(runner, 'rank', 0) != 0:
                return
            import numpy as np
            if len(self.iter_times) != self.measure_iters:
                raise RuntimeError(
                    'measured iteration count {} expected {}'.format(
                        len(self.iter_times), self.measure_iters))
            iter_array = np.asarray(self.iter_times, dtype=np.float64)
            data_array = np.asarray(self.data_times, dtype=np.float64)
            report = dict(
                schema_version=1,
                status='PASS',
                config=self.config_path,
                config_sha256=self.config_sha256,
                warmup_iters=self.warmup_iters,
                measured_iters=self.measure_iters,
                samples_per_gpu=self.samples_per_gpu,
                data_time_s=dict(
                    mean=float(data_array.mean()),
                    p50=percentile(data_array, 0.50),
                    p90=percentile(data_array, 0.90),
                    maximum=float(data_array.max())),
                iter_time_s=dict(
                    mean=float(iter_array.mean()),
                    p50=percentile(iter_array, 0.50),
                    p90=percentile(iter_array, 0.90),
                    maximum=float(iter_array.max())),
                throughput_samples_per_s=(
                    self.samples_per_gpu / float(iter_array.mean())),
                cuda_peak_memory_mib=(
                    torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)),
                cuda_peak_reserved_mib=(
                    torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)),
                raw=dict(
                    data_time_s=self.data_times,
                    iter_time_s=self.iter_times))
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                encoding='utf-8')
            print('N7_TRAINING_BENCHMARK=PASS output={} iter_mean={:.6f} '
                  'data_mean={:.6f} peak_mib={:.1f}'.format(
                      self.output_path,
                      report['iter_time_s']['mean'],
                      report['data_time_s']['mean'],
                      report['cuda_peak_memory_mib']))

    return N7TrainingBenchmarkHook


def run(args):
    # legacy 训练依赖延迟导入，保证现代本地容器仍可运行 --help。
    import mmcv
    import torch
    from mmcv import Config
    from mmdet.apis import set_random_seed
    from mmdet3d.apis import train_model
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    register_benchmark_hook()
    cfg = Config.fromfile(str(args.config))
    total_iters = args.warmup_iters + args.measure_iters
    cfg.runner = dict(type='IterBasedRunner', max_iters=total_iters)
    cfg.pop('total_epochs', None)
    cfg.workflow = [('train', 1)]
    cfg.work_dir = str(args.work_dir)
    cfg.gpu_ids = range(1)
    cfg.seed = args.seed
    cfg.resume_from = None
    if args.no_load:
        cfg.load_from = None
    cfg.checkpoint_config = None
    cfg.log_config = dict(
        interval=max(1, args.log_interval),
        hooks=[dict(type='TextLoggerHook')])
    if args.samples_per_gpu is not None:
        cfg.data.samples_per_gpu = int(args.samples_per_gpu)
    if args.workers_per_gpu is not None:
        cfg.data.workers_per_gpu = int(args.workers_per_gpu)
    output_path = args.output or (args.work_dir / 'training_benchmark.json')
    cfg.custom_hooks = [dict(
        type='N7TrainingBenchmarkHook',
        output_path=str(output_path),
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        samples_per_gpu=int(cfg.data.samples_per_gpu),
        config_path=str(args.config),
        config_sha256=file_sha256(args.config),
        priority='LOWEST')]

    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    cfg.dump(str(args.work_dir / 'resolved_benchmark_config.py'))
    set_random_seed(args.seed, deterministic=args.deterministic)
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    dataset = build_dataset(cfg.data.train)
    model.CLASSES = dataset.CLASSES
    train_model(
        model, [dataset], cfg,
        distributed=False,
        validate=False,
        timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
        meta=dict(
            seed=args.seed,
            exp_name=args.config.name,
            benchmark=True))
    if not output_path.is_file():
        raise RuntimeError('benchmark report was not written: {}'.format(output_path))


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config', type=Path)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--warmup-iters', type=int, default=10)
    parser.add_argument('--measure-iters', type=int, default=50)
    parser.add_argument('--samples-per-gpu', type=int)
    parser.add_argument('--workers-per-gpu', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--no-load', action='store_true')
    parser.add_argument('--log-interval', type=int, default=10)
    return parser


def main():
    args = build_parser().parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if args.warmup_iters < 0 or args.measure_iters <= 0:
        raise ValueError('warmup-iters must be >=0 and measure-iters must be positive')
    run(args)


if __name__ == '__main__':
    main()
