# -*- coding: utf-8 -*-
from genericpath import exists
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmseg.models import build_head as build_seg_head
from mmdet.models.detectors import BaseDetector
from mmdet3d.core import bbox3d2result
from mmseg.ops import resize
from mmcv.runner import get_dist_info, auto_fp16

import copy
import onnxruntime

from tools.utils import get_bboxes
import numpy as np
import sys
# import bstnnx

@DETECTORS.register_module()
class FastBEV(BaseDetector):
    # 添加类变量用于跟踪校准数据保存的迭代次数
    test_iteration_counter = 0

    def __init__(
        self,
        backbone,
        neck,
        neck_fuse,
        neck_3d,
        bbox_head,
        seg_head,
        n_voxels,
        voxel_size,
        bbox_head_2d=None,
        train_cfg=None,
        test_cfg=None,
        train_cfg_2d=None,
        test_cfg_2d=None,
        pretrained=None,
        init_cfg=None,
        extrinsic_noise=0,
        seq_detach=False,
        multi_scale_id=None,
        multi_scale_3d_scaler=None,
        with_cp=False,
        backproject='inplace',
        style='v4',
        # 可以添加校准数据保存相关的参数
        calibration_save_dir=None,
        # max_calibration_samples=75,
        max_calibration_samples=500,        
    ):
        super().__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.neck_3d = build_neck(neck_3d)
        if isinstance(neck_fuse['in_channels'], list):
            for i, (in_channels, out_channels) in enumerate(zip(neck_fuse['in_channels'], neck_fuse['out_channels'])):
                self.add_module(
                    f'neck_fuse_{i}', 
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        else:
            self.neck_fuse = nn.Conv2d(neck_fuse["in_channels"], neck_fuse["out_channels"], 3, 1, 1)
        
        # style
        # v1: fastbev wo/ ms
        # v2: fastbev + img ms
        # v3: fastbev + bev ms
        # v4: fastbev + img/bev ms
        self.style = style
        assert self.style in ['v1', 'v2', 'v3', 'v4'], self.style
        self.multi_scale_id = multi_scale_id
        self.multi_scale_3d_scaler = multi_scale_3d_scaler

        if bbox_head is not None:
            bbox_head.update(train_cfg=train_cfg)
            bbox_head.update(test_cfg=test_cfg)
            self.bbox_head = build_head(bbox_head)
            self.bbox_head.voxel_size = voxel_size
        else:
            self.bbox_head = None

        if seg_head is not None:
            self.seg_head = build_seg_head(seg_head)
        else:
            self.seg_head = None

        if bbox_head_2d is not None:
            bbox_head_2d.update(train_cfg=train_cfg_2d)
            bbox_head_2d.update(test_cfg=test_cfg_2d)
            self.bbox_head_2d = build_head(bbox_head_2d)
        else:
            self.bbox_head_2d = None

        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # test time extrinsic noise
        self.extrinsic_noise = extrinsic_noise
        if self.extrinsic_noise > 0:
            for i in range(5):
                print("### extrnsic noise: {} ###".format(self.extrinsic_noise))

        # detach adj feature
        self.seq_detach = seq_detach
        self.backproject = backproject
        # checkpoint
        self.with_cp = with_cp
        # 添加保存校准数据的配置
        self.calibration_save_dir = calibration_save_dir
        self.max_calibration_samples = max_calibration_samples        

    @staticmethod
    def _compute_projection(img_meta, stride, noise=0):
        projection = []
        intrinsic = torch.tensor(img_meta["lidar2img"]["intrinsic"][:3, :3])
        intrinsic[:2] /= stride
        extrinsics = map(torch.tensor, img_meta["lidar2img"]["extrinsic"])
        for extrinsic in extrinsics:
            if noise > 0:
                projection.append(intrinsic @ extrinsic[:3] + noise)
            else:
                projection.append(intrinsic @ extrinsic[:3])
        #   change 
        img_meta["lidar2img"]["intrinsic"][:3, :3]=intrinsic

        return torch.stack(projection)

    def extract_feat(self, img, img_metas, mode):
        ###################### 前处理 ######################
        batch_size = img.shape[0]
        img = img.reshape(
            [-1] + list(img.shape)[2:]
        )  # [1, 6, 3, 928, 1600] -> [6, 3, 928, 1600]
        ###################### 前处理 ######################

        ###################### backbone resnet + fpn ######################
        if mode in ['test', 'train']:
            x = self.backbone(
                img
            )  # [6, 256, 232, 400]; [6, 512, 116, 200]; [6, 1024, 58, 100]; [6, 2048, 29, 50]

            # use for vovnet
            if isinstance(x, dict):
                tmp = []
                for k in x.keys():
                    tmp.append(x[k])
                x = tmp

            # fuse features
            def _inner_forward(x):
                out = self.neck(x)
                return out  # old [6, 64, 232, 400]; [6, 64, 116, 200]; [6, 64, 58, 100]; [6, 64, 29, 50])
                            # [6*16, 64, 64, 176];[6*16, 64, 32, 88];[6*16, 64, 16, 44]; [6*16, 64, 8, 22]

            if self.with_cp and x.requires_grad:
                mlvl_feats = cp.checkpoint(_inner_forward, x)
            else:
                mlvl_feats = _inner_forward(x)
            mlvl_feats = list(mlvl_feats)

            features_2d = None
            if self.bbox_head_2d:
                features_2d = mlvl_feats

            if self.multi_scale_id is not None:
                mlvl_feats_ = []
                for msid in self.multi_scale_id:
                    # fpn output fusion
                    if getattr(self, f'neck_fuse_{msid}', None) is not None:
                        fuse_feats = [mlvl_feats[msid]]
                        for i in range(msid + 1, len(mlvl_feats)):    # 上采样
                            resized_feat = resize(
                                mlvl_feats[i], 
                                size=mlvl_feats[msid].size()[2:], 
                                mode="nearest",)
                            fuse_feats.append(resized_feat)
                    
                        if len(fuse_feats) > 1:
                            fuse_feats = torch.cat(fuse_feats, dim=1)
                        else:
                            fuse_feats = fuse_feats[0]
                        fuse_feats = getattr(self, f'neck_fuse_{msid}')(fuse_feats)    # 256 -> 64   多尺度融合
                        mlvl_feats_.append(fuse_feats)
                    else:
                        mlvl_feats_.append(mlvl_feats[msid])
            mlvl_feats = mlvl_feats_  # 24, 64, 64, 176 -> 每一帧的2d特征
                                    # 6*16, 64, 64, 176 -> 每一帧的2d特征

        elif mode == 'test_onnx':
            onnx_model_path = 'work_dirs/od_train_20251017_20251030_20251031_20251203_r18_gpu8_batch28_work_8_260407/fp_onnx/simplified_export_2d_model.onnx'  # TODO: change your onnx
            providers = ['CUDAExecutionProvider']
            sess = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
            # model = bstnnx.load(onnx_model_path)
            input_name = sess.get_inputs()[0].name
            all_outputs = []
            for i in range(img.size(0)):
                input_data_single = img[i].unsqueeze(0).cpu().float()  # 1,3,256,704
                # onnx_output = bstnnx.backend.CPUBackend.run_model(model,
                #                                                   input_data_single.numpy(),
                #                                                   custom_op_lib=bstnnx.backend.custom_op.get_custom_op_lib_path())
                # onnx_input = {'prep_input_input.1': input_data_single.numpy()}  # prep_input_input.1
                onnx_input = {input_name: input_data_single.numpy()}
                onnx_output = sess.run(None, onnx_input)
                output_data_single = torch.from_numpy(onnx_output[0]).to(img[0].device)
                all_outputs.append(output_data_single)
            mlvl_feats = [torch.cat(all_outputs, dim=0)]
        else:
            raise ValueError(f"Unsupported test mode: {mode}")
        ###################### backbone resnet + fpn ######################
        
        ###################### 2d -> 3d ######################
        # v3 bev ms
        if isinstance(self.n_voxels, list) and len(mlvl_feats) < len(self.n_voxels):
            pad_feats = len(self.n_voxels) - len(mlvl_feats)
            for _ in range(pad_feats):
                mlvl_feats.append(mlvl_feats[0])

        mlvl_volumes = []
        for lvl, mlvl_feat in enumerate(mlvl_feats):  # 6*16, 64, 64, 176 
            stride_i = math.ceil(img.shape[-1] / mlvl_feat.shape[-1])  # P4 880 / 32 = 27.5   new： 704/176  = 4   特征图缩小尺寸
            # [bs*seq*nv, c, h, w] -> [bs, seq*nv, c, h, w]
            mlvl_feat = mlvl_feat.reshape([batch_size, -1] + list(mlvl_feat.shape[1:]))  # bs,24,64,64,176
            # [bs, seq*nv, c, h, w] -> list([bs, nv, c, h, w])
            mlvl_feat_split = torch.split(mlvl_feat, 6, dim=1)  # bs, 6, 64, 64, 176 * 4(seq)  按每帧数据拆成list

            volume_list = []
            for seq_id in range(len(mlvl_feat_split)):
                volumes = []
                for batch_id, seq_img_meta in enumerate(img_metas):
                    feat_i = mlvl_feat_split[seq_id][batch_id]  # [nv, c, h, w]
                    img_meta = copy.deepcopy(seq_img_meta)
                    img_meta["lidar2img"]["extrinsic"] = img_meta["lidar2img"]["extrinsic"][seq_id*6:(seq_id+1)*6]  #取每个训练数据当前seq的六个相机外参 
                    if isinstance(img_meta["img_shape"], list):
                        img_meta["img_shape"] = img_meta["img_shape"][seq_id*6:(seq_id+1)*6]
                        img_meta["img_shape"] = img_meta["img_shape"][0]
                    height = math.ceil(img_meta["img_shape"][0] / stride_i)
                    width = math.ceil(img_meta["img_shape"][1] / stride_i)

                    projection = self._compute_projection(
                        img_meta, stride_i, noise=self.extrinsic_noise).to(feat_i.device)
                    if self.style in ['v1', 'v2']:
                        # wo/ bev ms
                        n_voxels, voxel_size = self.n_voxels[0], self.voxel_size[0]
                    else:
                        # v3/v4 bev ms
                        n_voxels, voxel_size = self.n_voxels[lvl], self.voxel_size[lvl]
                    points = get_points(  # [3, vx, vy, vz]  -> 3, 200, 200, 4 (数值 * 坐标)
                        n_voxels=torch.tensor(n_voxels),
                        voxel_size=torch.tensor(voxel_size),
                        origin=torch.tensor(img_meta["lidar2img"]["origin"]),
                    ).to(feat_i.device)

                    if self.backproject == 'inplace':
                        volume = backproject_inplace(
                            feat_i[:, :, :height, :width], points, projection,img_meta,seq_id)  # [c, vx, vy, vz]
                    else:
                        volume, valid = backproject_vanilla(
                            feat_i[:, :, :height, :width], points, projection)
                        volume = volume.sum(dim=0)
                        valid = valid.sum(dim=0)
                        volume = volume / valid
                        valid = valid > 0
                        volume[:, ~valid[0]] = 0.0
                    
                    # volumes.append(volume)
                    ########## change 1 ##############
                    volumes.append(volume.permute(3, 0, 1, 2).reshape(1, 256, 200, 200))  # 64, 200, 200, 4 -> 4, 64, 200, 200, -> 1, 256, 200, 200
                    ########## change 1 ##############
                volume_list.append(torch.stack(volumes))  # list([bs, c, vx, vy, vz])  64, 200, 200, 4 * 4 (single point feature * seq)
    
            mlvl_volumes.append(torch.cat(volume_list, dim=1))  # list([bs, seq*c, vx, vy, vz])
        
        if self.style in ['v1', 'v2']:
            mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, lvl*seq*c, vx, vy, vz]   #  4 4 256 200 200     list([bs, seq, vz*c ,vx, vy])
        else:
            # bev ms: multi-scale bev map (different x/y/z)
            for i in range(len(mlvl_volumes)):
                mlvl_volume = mlvl_volumes[i]
                bs, c, x, y, z = mlvl_volume.shape
                # collapse h, [bs, seq*c, vx, vy, vz] -> [bs, seq*c*vz, vx, vy]
                mlvl_volume = mlvl_volume.permute(0, 2, 3, 4, 1).reshape(bs, x, y, z*c).permute(0, 3, 1, 2)
                
                # different x/y, [bs, seq*c*vz, vx, vy] -> [bs, seq*c*vz, vx', vy']
                if self.multi_scale_3d_scaler == 'pool' and i != (len(mlvl_volumes) - 1):
                    # pooling to bottom level
                    mlvl_volume = F.adaptive_avg_pool2d(mlvl_volume, mlvl_volumes[-1].size()[2:4])
                elif self.multi_scale_3d_scaler == 'upsample' and i != 0:  
                    # upsampling to top level 
                    mlvl_volume = resize(
                        mlvl_volume,
                        mlvl_volumes[0].size()[2:4],
                        mode='bilinear',
                        align_corners=False)
                else:
                    # same x/y
                    pass

                # [bs, seq*c*vz, vx', vy'] -> [bs, seq*c*vz, vx, vy, 1]
                mlvl_volume = mlvl_volume.unsqueeze(-1)
                mlvl_volumes[i] = mlvl_volume
            mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, z1*c1+z2*c2+..., vx, vy, 1]
        ###################### 2d -> 3d ######################
        
        ###################### 3d neck ######################
        if mode in ['test', 'train']:
            x = mlvl_volumes

            # N, C*T, X, Y, Z -> N, X, Y, Z, C -> N, X, Y, Z*C*T -> N, Z*C*T, X, Y
            # N, C, X, Y, Z = x.shape
            ########## change 1 ##############
            N, Z, C, X, Y = x.shape
            x = x.reshape(N, Z*C, X, Y)
            ########## change 1 ##############

            def _inner_forward(x):
                # v1/v2: [bs, lvl*seq*c, vx, vy, vz] -> [bs, c', vx, vy]
                # v3/v4: [bs, z1*c1+z2*c2+..., vx, vy, 1] -> [bs, c', vx, vy]
                out = self.neck_3d(x)
                return out
                
            if self.with_cp and x.requires_grad:
                x = cp.checkpoint(_inner_forward, x)
            else:
                x = _inner_forward(x)

        elif mode == 'test_onnx':
            x_0 = mlvl_volumes[0][0].unsqueeze(0).cpu().float().numpy()
            x_1 = mlvl_volumes[0][1].unsqueeze(0).cpu().float().numpy()
            x_2 = mlvl_volumes[0][2].unsqueeze(0).cpu().float().numpy()
            x_3 = mlvl_volumes[0][3].unsqueeze(0).cpu().float().numpy()
            
            # ########### change liu.jiaren 20250318:新增保存一批input数据用作模型部署时的校验 ###########
            # 创建保存目录
            save_3d_quant_data = False
            save_base_dir = 'output/od_train_20251017_20251030_20251031_20251203_r18_gpu8_batch28_work_8_260407/3d_quant_data'
            os.makedirs(save_base_dir, exist_ok=True)
            
            if save_3d_quant_data and self.__class__.test_iteration_counter <= self.max_calibration_samples and self.__class__.test_iteration_counter % 5 == 0:
                print(f"正在保存迭代: {self.__class__.test_iteration_counter}")
                
                # 为每个输入创建子目录并保存数据
                for i in range(4):
                    # 创建子目录
                    sub_dir = os.path.join(save_base_dir, str(i))
                    os.makedirs(sub_dir, exist_ok=True)
                    
                    # 获取对应的输入数据
                    input_data = eval(f'x_{i}')
                    
                    # 保存为npy文件，文件名为当前迭代次数
                    np.save(os.path.join(sub_dir, f'{self.__class__.test_iteration_counter}.npy'), input_data)
                
                print(f"保存完成: {save_base_dir}")
                
            # 增加计数器，为下一次迭代做准备
            self.__class__.test_iteration_counter += 1
            # # ########### change liu.jiaren 20250318:新增保存一批input数据用作模型部署时的校验 ###########              

            onnx_model_path = 'work_dirs/od_train_20251017_20251030_20251031_20251203_r18_gpu8_batch28_work_8_260407/fp_onnx/simplified_export_3d_model.onnx'  # TODO: change your onnx
            # model = bstnnx.load(onnx_model_path)
            # input_data_single = x.cpu().float().numpy()  # 1,3,256,704
            providers = ['CUDAExecutionProvider']
            sess = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
            # input_name = sess.get_inputs()[0].name
            onnx_input = {'0': x_0, '1': x_1, '2': x_2, '3': x_3}
            # onnx_input = {'input.1': input_data_single}
            onnx_outputs = sess.run(None, onnx_input)
            # onnx_outputs = bstnnx.backend.CPUBackend.run_model(model,
            #                                                   input_data_single,
            #                                                   custom_op_lib=bstnnx.backend.custom_op.get_custom_op_lib_path())
            out1 = torch.from_numpy(onnx_outputs[0]).to(img[0].device)
            out2 = torch.from_numpy(onnx_outputs[1]).to(img[0].device)
            out3 = torch.from_numpy(onnx_outputs[2]).to(img[0].device)
            x = ([out1], [out2], [out3])
            features_2d = None
        else:
            raise ValueError(f"Unsupported test mode: {mode}")
        ###################### 3d neck ######################
            
        return x, None, features_2d

    def extract_feat_custom(self, img, img_metas, mode):
        ############################################ 前处理 ############################################
        batch_size = img.shape[0]
        img = img.reshape([-1] + list(img.shape)[2:])  # [1, 6, 3, 928, 1600] -> [6, 3, 928, 1600]
        ############################################ 前处理 ############################################

        ############################################ backbone resnet + fpn ############################################
        onnx_model_path = './output/quant_dir/2d_quant.onnx'  # TODO: change your onnx
        providers = ['CUDAExecutionProvider']
        sess = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
        # model = bstnnx.load(onnx_model_path)
        # input_name = sess.get_inputs()[0].name
        all_outputs = []
        for i in range(img.size(0)):
            input_data_single = img[i].unsqueeze(0).cpu().float()  # 1,3,256,704
            # onnx_output = bstnnx.backend.CPUBackend.run_model(model,
            #                                                   input_data_single.numpy(),
            #                                                   custom_op_lib=bstnnx.backend.custom_op.get_custom_op_lib_path())
            onnx_input = {'prep_input_input.1': input_data_single.numpy()}  # prep_input_input.1
            onnx_output = sess.run(None, onnx_input)
            output_data_single = torch.from_numpy(onnx_output[0]).to(img[0].device)
            all_outputs.append(output_data_single)
        mlvl_feats = [torch.cat(all_outputs, dim=0)]
        ############################################ backbone resnet + fpn ############################################
        
        ############################################ 2d -> 3d ############################################
        # v3 bev ms
        if isinstance(self.n_voxels, list) and len(mlvl_feats) < len(self.n_voxels):
            pad_feats = len(self.n_voxels) - len(mlvl_feats)
            for _ in range(pad_feats):
                mlvl_feats.append(mlvl_feats[0])

        mlvl_volumes = []
        for lvl, mlvl_feat in enumerate(mlvl_feats):  # 24,64,64,176
            stride_i = math.ceil(img.shape[-1] / mlvl_feat.shape[-1])  # P4 880 / 32 = 27.5
            # [bs*seq*nv, c, h, w] -> [bs, seq*nv, c, h, w]
            mlvl_feat = mlvl_feat.reshape([batch_size, -1] + list(mlvl_feat.shape[1:]))  # 1,24,64,64,176
            # [bs, seq*nv, c, h, w] -> list([bs, nv, c, h, w])
            mlvl_feat_split = torch.split(mlvl_feat, 6, dim=1)  # 1, 6, 64, 64, 176 * 4(seq)

            volume_list = []
            for seq_id in range(len(mlvl_feat_split)):
                volumes = []
                for batch_id, seq_img_meta in enumerate(img_metas):
                    feat_i = mlvl_feat_split[seq_id][batch_id]  # [nv, c, h, w]
                    img_meta = copy.deepcopy(seq_img_meta)
                    img_meta["lidar2img"]["extrinsic"] = img_meta["lidar2img"]["extrinsic"][seq_id*6:(seq_id+1)*6]
                    if isinstance(img_meta["img_shape"], list):
                        img_meta["img_shape"] = img_meta["img_shape"][seq_id*6:(seq_id+1)*6]
                        img_meta["img_shape"] = img_meta["img_shape"][0]
                    height = math.ceil(img_meta["img_shape"][0] / stride_i)
                    width = math.ceil(img_meta["img_shape"][1] / stride_i)

                    projection = self._compute_projection(
                        img_meta, stride_i, noise=self.extrinsic_noise).to(feat_i.device)
                    
                    # wo/ bev ms
                    n_voxels, voxel_size = self.n_voxels[0], self.voxel_size[0]

                    points = get_points(  # [3, vx, vy, vz]  -> 3, 200, 200, 4 (数值 * 坐标)
                        n_voxels=torch.tensor(n_voxels),
                        voxel_size=torch.tensor(voxel_size),
                        origin=torch.tensor(img_meta["lidar2img"]["origin"]),
                    ).to(feat_i.device)
                    
                    # ######################################## fix index!!! ########################################
                    # projection = torch.tensor(np.load("/workspace/fastbev/debug_gather/projection.npy")).cuda()  # fix index
                    # points = torch.tensor(np.load("/workspace/fastbev/debug_gather/points.npy")).cuda()
                    # ######################################## fix index!!! ########################################

                    volume = backproject_inplace(feat_i[:, :, :height, :width], points, projection)  # [c, vx, vy, vz]

                    # volumes.append(volume)  # 64, 200, 200, 4 (点云特征)
                    ########## change 1 ##############
                    volumes.append(volume.permute(3, 0, 1, 2).reshape(1, 256, 200, 200))  # 64, 200, 200, 4 -> 4, 64, 200, 200, -> 1, 256, 200, 200
                    ########## change 1 ##############
                volume_list.append(torch.stack(volumes))  # list([bs, c, vx, vy, vz])  64, 200, 200, 4 * 4 (single point feaature * seq)
    
            mlvl_volumes.append(torch.cat(volume_list, dim=1))  # list([bs, seq*c, vx, vy, vz])
        
        mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, lvl*seq*c, vx, vy, vz]
        ############################################ 2d -> 3d ############################################
        

        ############################################ 3d neck ############################################
        x_0 = mlvl_volumes[0][0].unsqueeze(0).cpu().float().numpy()
        x_1 = mlvl_volumes[0][1].unsqueeze(0).cpu().float().numpy()
        x_2 = mlvl_volumes[0][2].unsqueeze(0).cpu().float().numpy()
        x_3 = mlvl_volumes[0][3].unsqueeze(0).cpu().float().numpy()
        onnx_model_path = './output/quant_dir/3d_quant.onnx'  # TODO: change your onnx
        # model = bstnnx.load(onnx_model_path)
        # input_data_single = x.cpu().float().numpy()  # 1,3,256,704
        providers = ['CUDAExecutionProvider']
        sess = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
        # input_name = sess.get_inputs()[0].name
        # onnx_input = {'input.1': input_data_single}
        onnx_input = {'0': x_0, '1': x_1, '2': x_2, '3': x_3}
        onnx_outputs = sess.run(None, onnx_input)
        # onnx_outputs = bstnnx.backend.CPUBackend.run_model(model,
        #                                                   input_data_single,
        #                                                   custom_op_lib=bstnnx.backend.custom_op.get_custom_op_lib_path())
        out1 = torch.from_numpy(onnx_outputs[0]).to(img[0].device)
        out2 = torch.from_numpy(onnx_outputs[1]).to(img[0].device)
        out3 = torch.from_numpy(onnx_outputs[2]).to(img[0].device)
        x = ([out1], [out2], [out3])
        features_2d = None
        ############################################ 3d neck ############################################
            
        return x, None, features_2d

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            if img[0].shape == (1, 3, 256, 704):
                return self.onnx_export_2d(img[0], img_metas)  # backbone + neck + neck_fuse
            elif img[0].shape == (1, 256, 200, 200):
            # elif img.shape == (1, 1024, 200, 200):
                return self.onnx_export_3d(img, img_metas)  # neck_3d: 2d -> 3d
            else:
                raise NotImplementedError
        # if kwargs["export_onnx_flag"] == True:
        #     if kwargs["export_2d"]:
        #         return self.onnx_export_2d(img, img_metas)  # backbone + neck + neck_fuse
        #     elif kwargs["export_3d"]:
        #         return self.onnx_export_3d(img, img_metas)  # neck_3d: 2d -> 3d
        #     else:
        #         raise NotImplementedError

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(
        self, img, img_metas, gt_bboxes_3d, gt_labels_3d, gt_bev_seg=None, **kwargs
    ):
        feature_bev, valids, features_2d = self.extract_feat(img, img_metas, "train")
        """
        feature_bev: [(1, 256, 100, 100)]
        valids: (1, 1, 200, 200, 12)
        features_2d: [[6, 64, 232, 400], [6, 64, 116, 200], [6, 64, 58, 100], [6, 64, 29, 50]]
        """
        assert self.bbox_head is not None or self.seg_head is not None

        losses = dict()
        if self.bbox_head is not None:
            x = self.bbox_head(feature_bev)
            loss_det = self.bbox_head.loss(*x, gt_bboxes_3d, gt_labels_3d, img_metas)
            losses.update(loss_det)

        if self.seg_head is not None:
            assert len(gt_bev_seg) == 1
            x_bev = self.seg_head(feature_bev)
            gt_bev = gt_bev_seg[0][None, ...].long()
            loss_seg = self.seg_head.losses(x_bev, gt_bev)
            losses.update(loss_seg)

        if self.bbox_head_2d is not None:
            gt_bboxes = kwargs["gt_bboxes"][0]
            gt_labels = kwargs["gt_labels"][0]
            assert len(kwargs["gt_bboxes"]) == 1 and len(kwargs["gt_labels"]) == 1
            # hack a img_metas_2d
            img_metas_2d = []
            img_info = img_metas[0]["img_info"]
            for idx, info in enumerate(img_info):
                tmp_dict = dict(
                    filename=info["filename"],
                    ori_filename=info["filename"].split("/")[-1],
                    ori_shape=img_metas[0]["ori_shape"],
                    img_shape=img_metas[0]["img_shape"],
                    pad_shape=img_metas[0]["pad_shape"],
                    scale_factor=img_metas[0]["scale_factor"],
                    flip=False,
                    flip_direction=None,
                )
                img_metas_2d.append(tmp_dict)

            rank, world_size = get_dist_info()
            loss_2d = self.bbox_head_2d.forward_train(
                features_2d, img_metas_2d, gt_bboxes, gt_labels
            )
            losses.update(loss_2d)

        return losses

    def forward_test(self, img, img_metas, **kwargs):
        if not self.test_cfg.get('use_tta', False):
            if self.test_cfg.get('test_mode', False) == 'test':
                return self.simple_test(img, img_metas, mode='test')
            elif self.test_cfg.get('test_mode', False) == 'test_onnx':
                return self.simple_test(img, img_metas, mode='test_onnx')
            elif self.test_cfg.get('test_mode', False) == 'test_custom':
                return self.simple_test(img, img_metas, mode='test_custom')
            else:
                raise ValueError(f"Unsupported test mode: {self.test_cfg.get('test_mode', False)}")
        return self.aug_test(img, img_metas)

    def onnx_export_2d(self, img, img_metas):
        """
        input: 6, 3, 544, 960
        output: 6, 64, 136, 240
        """
        x = self.backbone(img)
        c1, c2, c3, c4 = self.neck(x)
        c2 = resize(
            c2, size=c1.size()[2:], mode="nearest"
        )  # [6, 64, 232, 400]
        c3 = resize(
            c3, size=c1.size()[2:], mode="nearest"
        )  # [6, 64, 232, 400]
        c4 = resize(
            c4, size=c1.size()[2:], mode="nearest"
        )  # [6, 64, 232, 400]
        x = torch.cat([c1, c2, c3, c4], dim=1)
        x = self.neck_fuse_0(x)

        if bool(os.getenv("DEPLOY", False)):
            x = x.permute(0, 2, 3, 1)
            return x

        return x

    def onnx_export_3d(self, x, _):
        # x: [6, 200, 100, 3, 256]
        # if bool(os.getenv("DEPLOY_DEBUG", False)):
        #     x = x.sum(dim=0, keepdim=True)
        #     return [x]
        x_0, x_1, x_2, x_3 = x  # 1, 256, 200, 200
        x = torch.cat((x_0, x_1, x_2, x_3), dim=1)  # 1, 1024, 200, 200

        if self.style == "v1":
            # x = x.sum(dim=0, keepdim=True)  # [1, 200, 100, 3, 256]
            x = self.neck_3d(x)  # [[1, 256, 100, 50], ]
        elif self.style == "v2":
            x = self.neck_3d(x)  # [6, 256, 100, 50]
            x = [x[0].sum(dim=0, keepdim=True)]  # [1, 256, 100, 50]
        elif self.style == "v3":
            x = self.neck_3d(x)  # [1, 256, 100, 50]
        else:
            raise NotImplementedError

        if self.bbox_head is not None:
            cls_score, bbox_pred, dir_cls_preds = self.bbox_head(x)
            # cls_score = [item.sigmoid() for item in cls_score]

        if dir_cls_preds is None:
            x = [cls_score, bbox_pred]
        else:
            x = [cls_score, bbox_pred, dir_cls_preds]
        # if os.getenv("DEPLOY", False):
        #     if dir_cls_preds is None:
        #         x = [cls_score, bbox_pred]
        #     else:
        #         x = [cls_score, bbox_pred, dir_cls_preds]
        #     return x

        return x

    def simple_test(self, img, img_metas, mode):
        bbox_results = []
        if mode in ['test', 'test_onnx']:
            feature_bev, _, features_2d = self.extract_feat(img, img_metas, mode)
        elif mode == 'test_custom':
            feature_bev, _, features_2d = self.extract_feat_custom(img, img_metas, mode)

        if self.bbox_head is not None:
            if self.test_cfg.get('test_mode', False) == 'test':
                x = self.bbox_head(feature_bev)
            elif self.test_cfg.get('test_mode', False) == 'test_onnx':
                x = feature_bev

            if self.test_cfg.get('test_mode', False) == 'test_custom':
                x = feature_bev
                bbox_list = get_bboxes(*x)
                # bbox_list_2 = self.bbox_head.get_bboxes(*x, img_metas, valid=None)
                bbox_results = [
                    bbox3d2result(det_bboxes, det_scores, det_labels)
                    for det_bboxes, det_scores, det_labels in bbox_list
                ]
                return bbox_results
            
            bbox_list = self.bbox_head.get_bboxes(*x, img_metas, valid=None)
            bbox_results = [
                bbox3d2result(det_bboxes, det_scores, det_labels)
                for det_bboxes, det_scores, det_labels in bbox_list
            ]

        else:
            bbox_results = [dict()]

        # BEV semantic seg
        if self.seg_head is not None:
            x_bev = self.seg_head(feature_bev)
            bbox_results[0]['bev_seg'] = x_bev

        return bbox_results

    def aug_test(self, imgs, img_metas):
        img_shape_copy = copy.deepcopy(img_metas[0]['img_shape'])
        extrinsic_copy = copy.deepcopy(img_metas[0]['lidar2img']['extrinsic'])

        x_list = []
        img_metas_list = []
        for tta_id in range(2):

            img_metas[0]['img_shape'] = img_shape_copy[24*tta_id:24*(tta_id+1)]
            img_metas[0]['lidar2img']['extrinsic'] = extrinsic_copy[24*tta_id:24*(tta_id+1)]
            img_metas_list.append(img_metas)

            feature_bev, _, _ = self.extract_feat(imgs[:, 24*tta_id:24*(tta_id+1)], img_metas, "test")
            x = self.bbox_head(feature_bev)
            x_list.append(x)

        bbox_list = self.bbox_head.get_tta_bboxes(x_list, img_metas_list, valid=None)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in [bbox_list]
        ]
        return bbox_results

    def show_results(self, *args, **kwargs):
        pass


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    points = torch.stack(
        torch.meshgrid(
            [
                torch.arange(n_voxels[0]),
                torch.arange(n_voxels[1]),
                torch.arange(n_voxels[2]),
            ],
            indexing='ij'
        )
    )
    new_origin = origin - n_voxels / 2.0 * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


def backproject_vanilla(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [6, 64, 200, 200, 12]
        valid: [6, 1, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]
    volume = torch.zeros(
        (n_images, n_channels, points.shape[-1]), device=features.device
    ).type_as(features)  # [6, 64, 480000]
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    # [6, 64, 480000] -> [6, 64, 200, 200, 12]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    # [6, 480000] -> [6, 1, 200, 200, 12]
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume, valid


def backproject_inplace(features, points, projection,img_mata,seq_id):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]  now [6, 64, 64, 176]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [64, 200, 200, 12]
    '''
    
    car_name = '9797_UKEF'
    save_path = f'output/2025_0418_2k_byd_info/index'
    save_npy_flag = False

    n_images, n_channels, height, width = features.shape  # 6, 64, 64, 176
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]  # 3, 200, 200, 4
    
    if save_npy_flag and seq_id == 0 and not os.path.exists(os.path.join(save_path, 'points.npy')):
        # 确保保存目录存在
        os.makedirs(save_path, exist_ok=True) 
        # 将输入张量保存为.npy文件 
        np.save(f"{save_path}/points.npy", points.detach().cpu().numpy())       
    
    # [3, 200, 200, 4] -> [1, 3, 160000] -> [6, 3, 160000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 160000] -> [6, 4, 160000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    
    # 输入为无畸变图片，源码中直接使用投影映射
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 160000] -> [6, 3, 160000]
    # points_2d_3 = torch.bmm(projection, points)  # lidar2img
    # x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 160000]
    # y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 160000]
    # z = points_2d_3[:, 2]  # [6, 160000]
    # valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 160000]
    
    # 考虑畸变-真值车系统
    fix_lut = True
    if not fix_lut: 
        points_2d_3, z_depth=get_points3d_from_pinhole_camera(img_mata, seq_id, points)

        x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 160000]
        y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 160000]
        z = points_2d_3[:, 2]  # [6, 160000]
        valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z_depth > 0)  # [6, 160000]
    else:
        # fix index
        x = torch.tensor(np.load(f"{save_path}/x.npy")).cuda()
        y = torch.tensor(np.load(f"{save_path}/y.npy")).cuda()
        valid = torch.tensor(np.load(f"{save_path}/valid.npy")).cuda()    

    # method2：特征填充，只填充有效特征，重复特征直接覆盖
    camera_list = [1, 2, 0, 4, 5, 3]   # 让前视占据更大的视野
    volume = torch.zeros(
        (n_channels, points.shape[-1]), device=features.device
    ).type_as(features)  # 64, 160000
    
    # for i in range(n_images):
    for i in camera_list:
        volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]    
    
    # lut可视化验证
    # test_vis_save_path = f'./work_dir/LUT_test_2k_9797_UKEF/'
    # test_valid_and_visualize(valid, seq_id, camera_list, test_vis_save_path) # 只在debug时候打开，正常训练中关闭
    
    if save_npy_flag and seq_id == 0 and not os.path.exists(os.path.join(save_path, 'valid.npy')):
        # 确保保存目录存在
        os.makedirs(save_path, exist_ok=True) 
        # 将输入张量保存为.npy文件
        np.save(f"{save_path}/x.npy", x.detach().cpu().numpy())
        np.save(f"{save_path}/y.npy", y.detach().cpu().numpy())  
        np.save(f"{save_path}/valid.npy", valid.detach().cpu().numpy())
        np.save(f"{save_path}/projection.npy", projection.detach().cpu().numpy()) 
        np.save(f"{save_path}/features.npy", features.detach().cpu().numpy())
        np.save(f"{save_path}/volume.npy", volume.detach().cpu().numpy())  
    
    # sys.exit()
    volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
           
    return volume

def get_points3d_from_pinhole_camera(img_meta,seq_id,points):  
    # 利用带畸变的相机内外参、畸变函数，生成像素——车辆坐标映射
    stride_intr=img_meta['lidar2img']['intrinsic']  # [4,4]
    extras=img_meta['lidar2img']['lidar2img_extra'][seq_id*6:(seq_id+1)*6]
    augs=img_meta['lidar2img']['lidar2img_aug'][seq_id*6:(seq_id+1)*6]
    final_points=[]
    z_cs=[]
    # 存在问题帧，该帧的内参中无distrotion参数，distrotion=[]，故统一使用最后一帧的
    # distrotion=img_meta['lidar2img']['lidar2img_extra'][-1]['distortion'] # [5] [k1,k2,p1,p2,k3]
    # distrotion=img_meta['lidar2img']['lidar2img_extra'][-1]['distortion'][0] # [5] [k1,k2,p1,p2,k3] # 适配2025_04_18_byd_info.json生成的pkl文件格式
    for i in range(len(extras)):
        intrinsic=augs[i]['intrin'] # [3,3]
        rot=augs[i]['rot'] # [3,3]
        tran=augs[i]['tran']  # [3]
        distrotion=extras[i]['distortion'][0] # [5] [k1,k2,p1,p2,k3]
        #加上各种变换后的内外参
        new_intrin=torch.tensor(stride_intr[:3,:3]@intrinsic).cuda()
        # 参考项目代码。。。
        lidar2cam_r = np.linalg.inv(rot)
        lidar2cam_t = tran @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t
        new_entrinsic=torch.tensor(lidar2cam_rt.T).cuda()
        # car -> cam
        point=points[i,:,:]
        cam_point=new_entrinsic.float()@point
        #引入相机畸变
        # xy_camera = cam_point[:2, :] / cam_point[ np.newaxis,2,:]
        z_c= cam_point[2,:].unsqueeze(0)
        xy_camera = cam_point[:2, :] /z_c
        x_c=xy_camera[0]
        y_c=xy_camera[1]
        r_squared = x_c**2 + y_c**2
        # 适配不同畸变参数
        if len(distrotion) == 5:
            k1, k2, p1, p2, _  = distrotion
            k3, k4, k5, k6 = 0, 0, 0, 0
        else:
            k1, k2, p1, p2, k3, k4, k5, k6 = distrotion

        # 畸变校正分子部分: 1 + k1*r² + k2*r⁴ + k3*r⁶
        radial_num = 1 + k1*r_squared + k2*r_squared**2 + k3*r_squared**3
        # 畸变校正分母部分: 1 + k4*r² + k5*r⁴ + k6*r⁶
        radial_den = 1 + k4*r_squared + k5*r_squared**2 + k6*r_squared**3 

        x_distorted = x_c * radial_num / radial_den + (2*p1*x_c*y_c + p2*(r_squared + 2*x_c**2))
        y_distorted = y_c * radial_num / radial_den + (p1*(r_squared + 2*y_c**2) + 2*p2*x_c*y_c)

        # cam -> img
        disr_points=torch.vstack((x_distorted, y_distorted,torch.ones(len(x_distorted)).cuda()))
        point_3d=new_intrin.float()@disr_points
        final_points.append(point_3d)
        z_cs.append(z_c)
    points_2d_3=torch.stack(final_points, dim=0)
    z_depth=torch.cat(z_cs, dim=0)
    return points_2d_3,z_depth

def test_valid_and_visualize(valid,seq_id, camera_list, output_dir):
    import cv2
    """
    测试 valid，并可视化最终有效的像素分布。

    """
    colors = [
        [255, 0, 0],    # 红色
        [0, 255, 0],    # 绿色
        [0, 0, 255],    # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 品红
        [0, 255, 255],  # 青色
    ]
    
    # 创建一个存储结果的颜色图像
    valid=valid.view(6,200, 200, 4).cpu().numpy()
    color_images = np.zeros((4, 200, 200, 3), dtype=np.uint8)   # 每张图片为 HxWx3 的颜色图像
    os.makedirs(output_dir+'single/', exist_ok=True) 
    #   单存相机视角范围
    for i in range(6):
        for j in range(4): 
            mask = valid[i, :, :, j]  # 当前相机对应维度 j 的有效位置
            mask = np.transpose(mask, (1, 0))  
            color_images_single=np.zeros((4, 200, 200, 3), dtype=np.uint8)
            color_images_single[j][mask] = colors[i]
            bgr_image_s = cv2.cvtColor(color_images_single[j], cv2.COLOR_RGB2BGR)
            save_path = os.path.join(output_dir+'single/', f"image_{seq_id+1}_{j+1}_{i+1}.png")
            cv2.imwrite(save_path, bgr_image_s)


    # 遍历每张图片，并根据 valid 提取信息
    # for i in range(6):
    for i in camera_list:
        for j in range(4):  # 最后一维的 4 个有效值
            mask = valid[i, :, :, j]  # 当前相机对应维度 j 的有效位置
            mask = np.transpose(mask, (1, 0))  
            # 将有效像素标记颜色
            color_images[j][mask] = colors[i]  # 将有效像素标记为对应的相机颜色
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录
    for j in range(4):
        bgr_image = cv2.cvtColor(color_images[j], cv2.COLOR_RGB2BGR)
        save_path = os.path.join(output_dir, f"image_{seq_id+1}_{j+1}.png")
        cv2.imwrite(save_path, bgr_image)
        #print(f"Image {j+1} saved to {output_dir}/image_{j+1}.png")
    


