# -*- coding: utf-8 -*-
"""N7 704x256 离线缓存图像训练配置。

这份配置用于 data/N7_704_256 数据根：
- 图片已经离线从 1600x900 强制 resize 到 704x256。
- info_json / pkl 中的相机内参仍保持 1600x900 标定，不要预先缩放 K。
- 新版 pkl 必须写入 intrinsic_width/height 和 image_width/height；
  RandomAugImageMultiViewImage 会按 intrinsic_* -> input_size 写入 post_rot，
  保证几何只缩放一次。
"""

_base_ = './custom_fastbev_6v_r18.py'

# 704x256 缓存数据根。converter 生成 pkl 时建议使用：
#   --data-path data/N7_704_256 --output-dir data/N7_704_256/pkl
# 这样 pkl 内图片路径相对 data/N7_704_256，训练时可直接解析。
data_root = './data/N7_704_256/'
ann_dir = data_root + 'pkl/'

# 默认值仅用于示例；正式训练时按 converter 输出文件名修改 ann_scope/ann_date，
# 或在训练脚本里用 --cfg-options 覆盖 ann_file。
ann_prefix = 'custom_fastbev'
ann_scope = '20251031_164821_1'
ann_date = '20260624'
ann_file_prefix = f'{ann_prefix}_{ann_scope}'

# 只覆盖数据根和 pkl 路径，模型、类别、时序、pipeline 继承 6V 基础配置。
data = dict(
    train=dict(
        data_root=data_root,
        ann_file=ann_dir + f'{ann_file_prefix}_infos_train_{ann_date}.pkl'),
    val=dict(
        data_root=data_root,
        ann_file=ann_dir + f'{ann_file_prefix}_infos_val_{ann_date}.pkl'),
    test=dict(
        data_root=data_root,
        ann_file=ann_dir + f'{ann_file_prefix}_infos_val_{ann_date}.pkl'))
