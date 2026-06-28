import mmcv
import torch
from mmcv.image import tensor2imgs
import os.path as osp
import pickle
import shutil
import tempfile
import time

from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import torch.distributed as dist
import ipdb


def _sync_cuda_for_profile():
    """同步 CUDA 队列，确保 profile 计时能覆盖真实 GPU 前向耗时。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _format_profile(prefix, iter_id, batch_size, data_time, forward_time,
                    post_time, iter_time, avg_data, avg_forward, avg_post,
                    avg_iter, sample_count):
    """格式化 test profile 输出，便于从日志直接判断瓶颈。"""
    throughput = batch_size / max(iter_time, 1e-6)
    avg_throughput = sample_count / max(avg_iter * max(iter_id + 1, 1), 1e-6)
    print(
        '{} #{:05d} batch={} data={:.3f}s forward={:.3f}s post={:.3f}s '
        'iter={:.3f}s task/s={:.2f} | avg data={:.3f}s forward={:.3f}s '
        'post={:.3f}s iter={:.3f}s task/s={:.2f}'.format(
            prefix, iter_id, batch_size, data_time, forward_time, post_time,
            iter_time, throughput, avg_data, avg_forward, avg_post, avg_iter,
            avg_throughput))


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    debug=False,
                    profile=False,
                    profile_interval=20):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    if debug:
        for i in range(5):
            print('#### debug mode in api/test.py, only 30 images ####')
    profile_interval = max(int(profile_interval), 1)
    total_data_time = 0.0
    total_forward_time = 0.0
    total_post_time = 0.0
    total_iter_time = 0.0
    total_samples = 0
    end = time.time()

    for i, data in enumerate(data_loader):
        data_time = time.time() - end
        if debug:
            if i > 30:
                return results

        if profile:
            _sync_cuda_for_profile()
        forward_start = time.time()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        if profile:
            _sync_cuda_for_profile()
        forward_time = time.time() - forward_start

        post_start = time.time()
        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(data, result, out_dir=out_dir)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'].data[0],
                                                  torch.Tensor):
                    img_tensor = data['img'].data[0][0]
                else:
                    raise NotImplementedError
                img_metas = data['img_metas'].data[0][0]

                imgs = tensor2imgs(img_tensor, **img_metas['img_norm_cfg'])
                assert len(imgs) == len(img_metas['img_info'])

                for j, (img, img_info) in enumerate(zip(imgs, img_metas['img_info'])):
                    h, w, _ = img_metas['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_metas['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_info['filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[j],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

        post_time = time.time() - post_start
        iter_time = time.time() - end
        total_data_time += data_time
        total_forward_time += forward_time
        total_post_time += post_time
        total_iter_time += iter_time
        total_samples += batch_size
        if profile and (i == 0 or (i + 1) % profile_interval == 0):
            denom = float(i + 1)
            _format_profile(
                '[TEST_PROFILE single]', i, batch_size, data_time,
                forward_time, post_time, iter_time, total_data_time / denom,
                total_forward_time / denom, total_post_time / denom,
                total_iter_time / denom, total_samples)
        end = time.time()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False,
                   debug=False, debug_num=50, profile=False,
                   profile_interval=20):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    if debug and world_size == 1:
        num_img = debug_num
        for i in range(5):
            print('#### debug mode in api/test.py, only {} images ####'.format(num_img))

    profile_interval = max(int(profile_interval), 1)
    total_data_time = 0.0
    total_forward_time = 0.0
    total_post_time = 0.0
    total_iter_time = 0.0
    total_samples = 0
    end = time.time()

    # ipdb.set_trace()
    for i, data in enumerate(data_loader):
        data_time = time.time() - end
        if debug and world_size == 1:
            if i > num_img:
                return results

        _sync_cuda_for_profile()
        forward_start = time.time()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            # if isinstance(result[0], tuple):
            #     result = [(bbox_results, encode_mask_results(mask_results))
            #               for bbox_results, mask_results in result]
        _sync_cuda_for_profile()
        forward_time = time.time() - forward_start

        post_start = time.time()
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

            post_time = time.time() - post_start
            iter_time = time.time() - end
            total_data_time += data_time
            total_forward_time += forward_time
            total_post_time += post_time
            total_iter_time += iter_time
            total_samples += batch_size * world_size
            if profile and (i == 0 or (i + 1) % profile_interval == 0):
                denom = float(i + 1)
                _format_profile(
                    '[TEST_PROFILE dist-rank0]', i, batch_size * world_size,
                    data_time, forward_time, post_time, iter_time,
                    total_data_time / denom, total_forward_time / denom,
                    total_post_time / denom, total_iter_time / denom,
                    total_samples)
        end = time.time()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
