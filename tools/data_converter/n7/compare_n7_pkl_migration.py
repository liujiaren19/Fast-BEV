#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""严格比较 N7 图片分辨率迁移前后的 Fast-BEV PKL。

该工具只读。它按 split/token 对齐旧、新 PKL，严格要求 token、clip、时间戳、
GT、相机顺序、K、畸变和外参不变；只允许图片路径、图片宽高及相应的顶层
转换元数据发生变化。同时保存 PKL/manifest SHA256，便于数据门禁归档。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


CAMERA_ARRAY_FIELDS = (
    'cam_intrinsic',
    'distortion',
    'sensor2lidar_rotation',
    'sensor2lidar_translation',
)
ALLOWED_INFO_KEYS = {'data_path', 'image_width', 'image_height'}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_payload(path: Path) -> Tuple[List[Dict], Dict]:
    with path.open('rb') as stream:
        payload = pickle.load(stream)
    if isinstance(payload, dict) and 'infos' in payload:
        return list(payload['infos']), dict(payload.get('metadata', {}) or {})
    if isinstance(payload, list):
        return list(payload), {}
    raise TypeError('unsupported pkl payload in {}: {}'.format(path, type(payload)))


def info_clip_key(info: Mapping) -> Tuple[str, str, str]:
    return tuple(str(info.get(key, '')) for key in ('dataset', 'sequence', 'clip'))


def scalar_equal(left, right) -> bool:
    if isinstance(left, np.generic):
        left = left.item()
    if isinstance(right, np.generic):
        right = right.item()
    return left == right


def array_equal(left, right) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    return (
        left_array.shape == right_array.shape and
        left_array.dtype == right_array.dtype and
        np.array_equal(left_array, right_array)
    )


def first_non_allowed_difference(left, right, path='info'):
    """返回忽略图片路径/尺寸后的首个语义差异路径。"""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_keys = set(left) - ALLOWED_INFO_KEYS
        right_keys = set(right) - ALLOWED_INFO_KEYS
        if left_keys != right_keys:
            return '{}.keys'.format(path)
        for key in sorted(left_keys):
            difference = first_non_allowed_difference(
                left[key], right[key], '{}.{}'.format(path, key))
            if difference is not None:
                return difference
        return None
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return '{}.length'.format(path)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = first_non_allowed_difference(
                left_item, right_item, '{}[{}]'.format(path, index))
            if difference is not None:
                return difference
        return None
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return None if array_equal(left, right) else path
    return None if scalar_equal(left, right) else path


def compare_camera(
    split: str,
    token: str,
    camera_id: str,
    old_cam: Mapping,
    new_cam: Mapping,
    expected_new_size: Tuple[int, int],
    failures: List[str],
    path_changes: Counter,
) -> None:
    prefix = '{} token={} camera={}'.format(split, token, camera_id)
    for field in CAMERA_ARRAY_FIELDS:
        if field not in old_cam or field not in new_cam:
            failures.append('{} missing calibration field {}'.format(prefix, field))
        elif not array_equal(old_cam[field], new_cam[field]):
            failures.append('{} changed calibration field {}'.format(prefix, field))

    for field in ('intrinsic_width', 'intrinsic_height'):
        if not scalar_equal(old_cam.get(field), new_cam.get(field)):
            failures.append('{} changed {}'.format(prefix, field))

    expected_h, expected_w = expected_new_size
    if int(new_cam.get('image_height', -1)) != expected_h:
        failures.append('{} image_height={} expected={}'.format(
            prefix, new_cam.get('image_height'), expected_h))
    if int(new_cam.get('image_width', -1)) != expected_w:
        failures.append('{} image_width={} expected={}'.format(
            prefix, new_cam.get('image_width'), expected_w))

    old_path = str(old_cam.get('data_path', ''))
    new_path = str(new_cam.get('data_path', ''))
    path_changes['changed' if old_path != new_path else 'unchanged'] += 1


def compare_split(
    split: str,
    old_path: Path,
    new_path: Path,
    expected_new_size: Tuple[int, int],
    max_examples: int,
) -> Dict:
    old_infos, old_metadata = load_payload(old_path)
    new_infos, new_metadata = load_payload(new_path)
    failures: List[str] = []
    path_changes = Counter()

    def token_map(infos: Sequence[Mapping], side: str) -> Dict[str, Mapping]:
        tokens = [str(info.get('token', '')) for info in infos]
        empty = sum(not token for token in tokens)
        duplicates = [token for token, count in Counter(tokens).items() if token and count > 1]
        if empty:
            failures.append('{} {} has {} empty tokens'.format(split, side, empty))
        if duplicates:
            failures.append('{} {} duplicate tokens {}'.format(
                split, side, duplicates[:max_examples]))
        return {str(info.get('token')): info for info in infos if info.get('token')}

    old_by_token = token_map(old_infos, 'old')
    new_by_token = token_map(new_infos, 'new')
    old_token_order = [str(info.get('token')) for info in old_infos]
    new_token_order = [str(info.get('token')) for info in new_infos]
    if old_token_order != new_token_order:
        failures.append('{} token order changed'.format(split))
    old_tokens = set(old_by_token)
    new_tokens = set(new_by_token)
    missing = sorted(old_tokens - new_tokens)
    extra = sorted(new_tokens - old_tokens)
    if missing:
        failures.append('{} missing new tokens {}'.format(split, missing[:max_examples]))
    if extra:
        failures.append('{} extra new tokens {}'.format(split, extra[:max_examples]))

    changed = Counter()
    for token in sorted(old_tokens & new_tokens):
        old_info = old_by_token[token]
        new_info = new_by_token[token]
        if info_clip_key(old_info) != info_clip_key(new_info):
            failures.append('{} token={} changed clip {} -> {}'.format(
                split, token, info_clip_key(old_info), info_clip_key(new_info)))
            changed['clip'] += 1
        if not scalar_equal(old_info.get('timestamp'), new_info.get('timestamp')):
            failures.append('{} token={} changed timestamp'.format(split, token))
            changed['timestamp'] += 1
        for field in ('gt_boxes', 'gt_names'):
            if not array_equal(old_info.get(field, []), new_info.get(field, [])):
                failures.append('{} token={} changed {}'.format(split, token, field))
                changed[field] += 1

        old_cams = old_info.get('cams', {}) or {}
        new_cams = new_info.get('cams', {}) or {}
        if list(old_cams) != list(new_cams):
            failures.append('{} token={} changed camera order {} -> {}'.format(
                split, token, list(old_cams), list(new_cams)))
            changed['camera_order'] += 1
            continue
        for camera_id in old_cams:
            compare_camera(
                split, token, camera_id,
                old_cams[camera_id], new_cams[camera_id],
                expected_new_size, failures, path_changes)
        semantic_difference = first_non_allowed_difference(old_info, new_info)
        if semantic_difference is not None:
            failures.append('{} token={} changed non-allowed field {}'.format(
                split, token, semantic_difference))
            changed['other_info'] += 1

    old_set = str(old_metadata.get('set', split))
    new_set = str(new_metadata.get('set', split))
    if old_set != split or new_set != split:
        failures.append('{} metadata set mismatch old={} new={}'.format(
            split, old_set, new_set))

    return dict(
        split=split,
        old_pkl=str(old_path),
        new_pkl=str(new_path),
        old_sha256=file_sha256(old_path),
        new_sha256=file_sha256(new_path),
        old_infos=len(old_infos),
        new_infos=len(new_infos),
        token_set_equal=old_tokens == new_tokens,
        token_order_equal=old_token_order == new_token_order,
        missing_tokens=len(missing),
        extra_tokens=len(extra),
        clip_sets_equal=(
            {info_clip_key(info) for info in old_infos} ==
            {info_clip_key(info) for info in new_infos}),
        changed_critical_fields=dict(changed),
        image_path_changes=dict(path_changes),
        metadata=dict(
            old_set=old_set,
            new_set=new_set,
            old_manifest_sources=list(old_metadata.get('manifest_sources', []) or []),
            new_manifest_sources=list(new_metadata.get('manifest_sources', []) or []),
            old_gt_box_origin=old_metadata.get('gt_box_origin'),
            new_gt_box_origin=new_metadata.get('gt_box_origin'),
        ),
        status='PASS' if not failures else 'FAIL',
        failures=failures[:max_examples],
        failure_count=len(failures),
    )


def manifest_records(paths: Iterable[Path]) -> List[Dict]:
    records = []
    for path in paths:
        lines = [
            line.strip() for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip() and not line.lstrip().startswith('#')
        ]
        records.append(dict(
            path=str(path),
            sha256=file_sha256(path),
            entries=len(lines),
        ))
    return records


def write_report(report: Mapping, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'migration_compare.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    assets = dict(
        pkls=[
            dict(path=item[key], sha256=item[key.replace('pkl', 'sha256')])
            for item in report['splits']
            for key in ('old_pkl', 'new_pkl')
        ],
        manifests=report['manifests'],
    )
    (output_dir / 'asset_hashes.json').write_text(
        json.dumps(assets, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')

    lines = [
        '# N7 1600x900 PKL 迁移对比',
        '',
        '- status: `{}`'.format(report['status']),
        '- test split: `test=val`（当前未虚构独立 test）',
        '',
        '| split | old/new infos | token | clip | critical changes | path changed/unchanged | status |',
        '| --- | ---: | --- | --- | ---: | ---: | --- |',
    ]
    for item in report['splits']:
        path_counts = item['image_path_changes']
        lines.append(
            '| {split} | {old_infos}/{new_infos} | {token} | {clip} | {critical} | {changed}/{unchanged} | {status} |'.format(
                split=item['split'],
                old_infos=item['old_infos'],
                new_infos=item['new_infos'],
                token='same' if item['token_set_equal'] else 'DIFF',
                clip='same' if item['clip_sets_equal'] else 'DIFF',
                critical=sum(item['changed_critical_fields'].values()),
                changed=path_counts.get('changed', 0),
                unchanged=path_counts.get('unchanged', 0),
                status=item['status']))
    if report['failures']:
        lines.extend(['', '## Failures', ''])
        lines.extend('- {}'.format(message) for message in report['failures'])
    (output_dir / 'migration_compare.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--old-train', type=Path, required=True)
    parser.add_argument('--old-val', type=Path, required=True)
    parser.add_argument('--new-train', type=Path, required=True)
    parser.add_argument('--new-val', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, nargs='*', default=[])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-new-image-size', type=int, nargs=2,
                        metavar=('HEIGHT', 'WIDTH'), default=(900, 1600))
    parser.add_argument('--max-examples', type=int, default=50)
    parser.add_argument('--strict', action='store_true')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for path in (
            args.old_train, args.old_val, args.new_train, args.new_val,
            *args.manifests):
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_size = tuple(int(value) for value in args.expected_new_image_size)
    split_reports = [
        compare_split(
            'train', args.old_train, args.new_train,
            expected_size, args.max_examples),
        compare_split(
            'val', args.old_val, args.new_val,
            expected_size, args.max_examples),
    ]
    cross_overlap = sorted(
        set(str(info.get('token')) for info in load_payload(args.new_train)[0]) &
        set(str(info.get('token')) for info in load_payload(args.new_val)[0]))
    failures = [
        message
        for split_report in split_reports
        for message in split_report['failures']
    ]
    if cross_overlap:
        failures.append('new train/val token overlap: {}'.format(
            cross_overlap[:args.max_examples]))
    report = dict(
        schema_version=1,
        status='PASS' if not failures else 'FAIL',
        allowed_changes=[
            'cams.*.data_path',
            'cams.*.image_width',
            'cams.*.image_height',
            'corresponding converter metadata and provenance paths',
        ],
        splits=split_reports,
        new_train_val_token_overlap=len(cross_overlap),
        manifests=manifest_records(args.manifests),
        failures=failures[:args.max_examples],
        failure_count=len(failures),
    )
    write_report(report, args.output_dir)
    print('N7_PKL_MIGRATION_COMPARE={} failures={}'.format(
        report['status'], report['failure_count']))
    if args.strict and report['status'] != 'PASS':
        sys.exit(2)


if __name__ == '__main__':
    main()
