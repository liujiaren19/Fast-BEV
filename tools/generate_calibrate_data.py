import argparse
import mmcv
import os
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from os import path as osp

mmdet3d_root = os.environ.get('MMDET3D')
if mmdet3d_root is not None and osp.exists(mmdet3d_root):
    import sys
    sys.path.insert(0, mmdet3d_root)
    print(f"using mmdet3d: {mmdet3d_root}")

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed


"""
Fast-BEV 量化校准数据导出脚本。

用途：
1. 使用 PyTorch checkpoint 跑 ``test_pth`` 推理链路。
2. 为板端 2D backbone 校准拷贝原始多视角图片。
3. 为板端 3D head 校准保存 detector 内部生成的 4D BEV 输入 npy。

推荐用法：

    python tools/generate_calibrate_data.py \
        --config configs/fastbev/custom/custom_fastbev_6v_r18_n7_704x256.py \
        --checkpoint work_dirs/n7_6v/epoch_20.pth \
        --calibration-dir compiler/n7_6v_704x256_epoch20 \
        --max-calibration-samples 500 \
        --workers-per-gpu 4

快速检查路径和 shape：

    python tools/generate_calibrate_data.py \
        --config configs/fastbev/custom/custom_fastbev_6v_r18_n7_704x256.py \
        --checkpoint work_dirs/n7_6v/epoch_2.pth \
        --calibration-dir compiler/debug_calib \
        --max-calibration-samples 20 \
        --debug --debug-num 3

输出目录：

    <calibration-dir>/backbone_calibrate_data/
        拷贝每个 BEV sample 对应的 ``6 views * 4 times`` 图片。

    <calibration-dir>/head_calibrate_data/0/
    <calibration-dir>/head_calibrate_data/1/
    <calibration-dir>/head_calibrate_data/2/
    <calibration-dir>/head_calibrate_data/3/
        保存 v1/R18 板端 3D head 的 4 个时序输入 npy，
        单个 npy 形状应为 ``[1, z*c, x, y]``。

本脚本和通用 ``tools/test.py`` 的关键区别：
- 强制 ``samples_per_gpu=1``。量化校准按单个 BEV sample 保存，避免把
  batch 维混进 head npy，导致量化工具统计到错误的激活分布。
- 不执行 ``format_results``、``evaluate``、``mmcv.dump(outputs)``。
  校准只需要中间输入文件，不需要测试结果 pkl 或提交格式。
- 只支持单进程 ``--launcher none``。多进程同时写同一 calibration-dir 会
  产生文件覆盖和样本编号冲突。
- 通过 ``cfg.model.test_cfg`` 打开 FastBEV 内部保存开关，不修改
  ``mmdet3d/apis/test.py`` 的通用测试逻辑。
"""


def parse_args():
    """解析量化校准专用参数。

    这里保留 ``--cfg-options`` 是为了临时覆盖 pkl 路径、data_root、
    ``model.use_distortion`` 等配置；删除测试脚本中的 ``--out``、
    ``--eval``、``--format-only``，避免校准任务误走评测/格式化分支。
    """
    parser = argparse.ArgumentParser(
        description='Generate Fast-BEV quantization calibration data')
    parser.add_argument(
        '--config',
        default='./configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py',
        help='calibration config file path')
    parser.add_argument(
        '--checkpoint',
        default='ckpts/epoch15_batch4_lr0002.pth',
        help='checkpoint file')
    parser.add_argument(
        '--calibration-dir',
        default='compiler',
        help='directory for backbone_calibrate_data and head_calibrate_data')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher; calibration export currently supports none only')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--debug',
        action='store_true',
        help='only run a few samples for path validation')
    parser.add_argument('--debug-num', type=int, default=10)
    parser.add_argument(
        '--workers-per-gpu',
        type=int,
        default=None,
        help='override cfg.data.workers_per_gpu for calibration export')
    parser.add_argument(
        '--max-calibration-samples',
        type=int,
        default=500,
        help='maximum BEV samples saved for backbone/head calibration data')
    parser.add_argument('--extrinsic-noise', '-n', type=float, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.max_calibration_samples <= 0:
        raise ValueError('--max-calibration-samples must be positive')
    if args.debug_num <= 0:
        raise ValueError('--debug-num must be positive')
    return args


def prepare_single_sample_test_cfg(data_test):
    """把 ``cfg.data.test`` 改成校准安全的单样本 dataloader 配置。

    原训练/测试配置中可能设置 ``samples_per_gpu=24`` 来提升吞吐；但校准
    数据保存逻辑的单位是一个 BEV sample。强制 batch=1 后：
    - backbone 侧一次只拷贝当前 sample 的 24 张时序多视角图片；
    - head 侧一次只保存当前 sample 的 4 个时序 BEV 输入；
    - ``calibrate_data_id`` 与输出文件一一对应。
    """
    original_samples = []
    if isinstance(data_test, dict):
        data_test.test_mode = True
        original_samples.append(data_test.pop('samples_per_gpu', 1))
    elif isinstance(data_test, list):
        for ds_cfg in data_test:
            ds_cfg.test_mode = True
            original_samples.append(ds_cfg.pop('samples_per_gpu', 1))
    else:
        raise TypeError('cfg.data.test must be a dict or list, got {}'.format(
            type(data_test)))

    original_samples = [int(x) for x in original_samples]
    if any(x != 1 for x in original_samples):
        print('Calibration export forces samples_per_gpu=1; '
              'original test samples_per_gpu={}'.format(original_samples))
    return 1


def run_calibration(model, data_loader, max_samples, debug=False, debug_num=10):
    """运行 pth 推理链路并只保存量化校准输入，不收集测试结果。

    ``FastBEV.extract_feat`` 会在前向过程中检查
    ``test_cfg['save_calibrate_data_flag']``，然后分别保存 backbone 和
    head 校准输入。这里不关心模型预测输出，只用 `_calibration_samples_saved`
    判断是否已经达到目标样本数。
    """
    model.eval()
    module = model.module if hasattr(model, 'module') else model
    target = min(len(data_loader.dataset), max_samples)
    if debug:
        target = min(target, debug_num)
        print('#### debug mode in generate_calibrate_data.py, only {} samples ####'.format(target))
    prog_bar = mmcv.ProgressBar(target)

    saved = int(getattr(module, '_calibration_samples_saved', 0))
    for i, data in enumerate(data_loader):
        # debug 模式用于验证路径、shape 和权限，不追求达到 max_samples。
        if debug and i >= debug_num:
            break
        if saved >= max_samples:
            break

        # 显式写入当前样本编号，让 backbone/head 的文件名可以对齐。
        if module.test_cfg is None:
            module.test_cfg = dict()
        module.test_cfg['calibrate_data_id'] = saved
        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        new_saved = int(getattr(module, '_calibration_samples_saved', saved))
        for _ in range(max(0, min(new_saved, target) - min(saved, target))):
            prog_bar.update()
        saved = new_saved

    print('\nSaved {} calibration samples.'.format(saved))
    if not debug and saved < min(max_samples, len(data_loader.dataset)):
        print('Warning: saved samples are fewer than requested. '
              'Check model style and calibration save logic.')
    return saved


def main():
    args = parse_args()

    if args.launcher != 'none':
        raise ValueError(
            'generate_calibrate_data.py writes shared calibration files and '
            'currently supports single-process --launcher none only.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    samples_per_gpu = prepare_single_sample_test_cfg(cfg.data.test)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    workers_per_gpu = cfg.data.get('workers_per_gpu', 1)
    if args.workers_per_gpu is not None:
        workers_per_gpu = args.workers_per_gpu
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        dist=False,
        shuffle=False)

    if args.extrinsic_noise > 0:
        for i in range(3):
            print('### test camera extrinsic robustness ###')
        cfg.model.extrinsic_noise = args.extrinsic_noise

    # 使用 pth 推理链路生成板端量化校准数据。这里不做 format/eval/out，
    # 因为量化校准只需要保存 backbone 原图输入和 3D head 的 4D BEV 输入。
    if cfg.model.get('test_cfg', None) is None:
        cfg.model.test_cfg = dict()
    cfg.model.test_cfg["test_mode"] = "test_pth"
    cfg.model.test_cfg["save_calibrate_data_flag"] = True
    calibration_dir = osp.abspath(args.calibration_dir)
    backbone_data_path = osp.join(calibration_dir, "backbone_calibrate_data")
    head_data_path = osp.join(calibration_dir, "head_calibrate_data")
    os.makedirs(backbone_data_path, exist_ok=True)
    os.makedirs(head_data_path, exist_ok=True)
    cfg.model.test_cfg["backbone_data_path"] = backbone_data_path
    cfg.model.test_cfg["head_data_path"] = head_data_path
    cfg.model["max_calibration_samples"] = args.max_calibration_samples
    # 校准数据按 FP32 保存，避免 fp16 hook 改变离线量化统计输入。
    # 这里直接置空 cfg.fp16，不调用 wrap_fp16_model。
    cfg.fp16 = None

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    model = MMDataParallel(model, device_ids=[0])
    run_calibration(
        model,
        data_loader,
        max_samples=args.max_calibration_samples,
        debug=args.debug,
        debug_num=args.debug_num)
    print('Backbone calibration data: {}'.format(backbone_data_path))
    print('Head calibration data: {}'.format(head_data_path))


if __name__ == '__main__':
    main()
