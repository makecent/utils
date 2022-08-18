# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
import numpy as np

from os import path as osp
import torch
from mmcv import Config, DictAction, mkdir_or_exist
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model

from mmaction.apis import single_gpu_test, multi_gpu_test
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.utils import register_module_hooks


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 clip-level feature extraction')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--out-dir',
        default=None,
        help='output result file into the specified folder')
    parser.add_argument('--video-list', help='video file list')
    parser.add_argument('--video-root', help='video root directory')
    parser.add_argument(
        '--clip-interval',
        default=4,
        help='interval among clips')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. For example, '
             "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    #%% Parse configuration
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg.merge_from_dict(args.cfg_options)

    if cfg.model.get('test_cfg'):
        cfg.model['test_cfg']['feature_extraction'] = True
    else:
        cfg.model['test_cfg'] = dict(feature_extraction=True)

    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, world_size = get_dist_info()

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    cfg.data.test.type = 'DenseExtracting'
    cfg.data.test.clip_interval = args.clip_interval
    if args.video_root:
        cfg.data.test.data_prefix = args.video_root
    if args.video_list:
        cfg.data.test.ann_file = args.video_list
    if args.checkpoint:
        cfg.load_from = args.checkpoint
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if cfg.out_dir is not None:
        mkdir_or_exist(cfg.out_dir)
    if cfg.out_dir is None:
        warnings.warn("Output directory was not specified, so dry-run only")

    #%% Init model
    # build the model and load checkpoint
    model = build_model(cfg.model)

    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    if cfg.load_from is not None:
        load_checkpoint(model, cfg.load_from, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

    #%% Init datasets and test
    ann_file = [x.strip() for x in open(cfg.data.test.ann_file).readlines()]
    for ann_line in ann_file:
        cfg.data.test.ann_file = ann_line
        dataset = build_dataset(cfg.data.test)
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            dist=distributed,
            shuffle=False)

        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('test_dataloader', {}))
        data_loader = build_dataloader(dataset, **dataloader_setting)

        if distributed:
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                     args.gpu_collect)
        else:
            outputs = single_gpu_test(model, data_loader)

        if cfg.out_dir and rank == 0:
            out_path = osp.join(args.out_dir, f'{dataset.video_name}.npy')
            print(f'\nwriting results to {out_path}')
            np.save(out_path, np.array(outputs))


if __name__ == '__main__':
    main()
