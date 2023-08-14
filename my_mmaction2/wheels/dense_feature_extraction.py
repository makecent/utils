# Execution example code:
# PYTHONPATH=$PWD:$PYTHONPATH bash my_mmaction2/wheels/dist_dense_feature_extraction.sh /home/louis/PycharmProjects/TAD_DINO/my_data/thumos14/rawframes/val 2  --out-dir /home/louis/PycharmProjects/TAD_DINO/my_data/thumos14/features/my_i3d_feature_v2/val_flow --format rawframes --modality Flow
import argparse
import os
from os import path as osp
from pathlib import Path

from mmengine import Config, DictAction, mkdir_or_exist
from mmengine.runner import Runner

from my_mmaction2.custom_modules import *


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 clip-level feature extraction')
    parser.add_argument('video_path', help='the path to the video files, could be raw videos or folders of raw frames')
    parser.add_argument('--format', choices=['video', 'rawframes'], default='video',
                        help='format of video files in video path, raw videos or pre-decoded raw frames')
    parser.add_argument('--modality', choices=['RGB', 'Flow'], default='RGB',
                        help='the format of feature to be extracted')
    parser.add_argument('--filename-tmpl', default=None,
                        help='the file name template of raw frames, not used for videos')
    parser.add_argument('--start-index', default=0,
                        help='the start index of raw frames, not used for videos')
    parser.add_argument(
        '--out-dir',
        default=None,
        help='output result file into the specified folder')
    parser.add_argument(
        '--clip-len',
        default=16,
        type=int,
        help='the number of frames of each clip')
    parser.add_argument(
        '--frame-interval',
        default=1,
        type=int,
        help='the frame interval between frames in a clip')
    parser.add_argument(
        '--clip-interval',
        default=4,
        type=int,
        help='the frame interval between clips')
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
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def get_cfg(args, video_name):
    cfg = Config()
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.launcher = args.launcher

    # ------------------- model settings (I3D) ----------------------- #
    # You may modify this part to use other models
    cfg.model = dict(
        type='FeatExtractor',
        data_preprocessor=dict(
            type='ActionDataPreprocessor',
            mean=[127.5, 127.5, 127.5] if args.modality == 'RGB' else [127.5, 127.5],
            std=[127.5, 127.5, 127.5] if args.modality == 'RGB' else [127.5, 127.5],
            format_shape='NCTHW'),
        backbone=dict(type='I3D', modality=f'{args.modality.lower()}'),
        neck=[dict(type='AdaptiveAvgPool3d', output_size=(1, 1, 1)),
              dict(type='Flatten', start_dim=1)]
    )
    cfg.load_from = f'https://github.com/makecent/utils/releases/download/v0.1/model_{args.modality.lower()}_backbone-prefix.pth'

    # ----------------------- data settings ------------------------- #
    if args.format == 'video':
        pipeline = [dict(type='DecordInit'), dict(type='DecordDecode')]
    else:
        pipeline = [dict(type='RawFrameDecode')]

    if args.filename_tmpl is None and args.format == 'rawframes':
        if args.modality == 'RGB':
            args.filename_tmpl = 'img_{:05}.jpg'
        else:
            args.filename_tmpl = 'flow_{}_{:05}.jpg'
    pipeline.extend([dict(type='Resize', scale=(-1, 256)),
                     # dict(type='CenterCrop', crop_size=224),
                     dict(type='TenCrop', crop_size=224),  # TenCrop as testing augmentation
                     dict(type='FormatShape', input_format='NCTHW'),
                     dict(type='PackActionInputs')])
    cfg.test_dataloader = dict(
        batch_size=1,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='DenseExtracting',
            video_name=video_name,
            pipeline=pipeline,
            clip_len=args.clip_len,
            frame_interval=args.frame_interval,
            clip_interval=args.clip_interval,
            data_prefix=args.video_path,
            filename_tmpl=args.filename_tmpl,
            start_index=args.start_index,
            vformat=args.format,
            modality=args.modality))

    # -------------------- Dump predictions --------------------
    if args.out_dir is not None:
        mkdir_or_exist(args.out_dir)
        cfg.work_dir = osp.join(args.out_dir, 'logs')
    else:
        cfg.work_dir = 'work_dir'
        warnings.warn("Output directory was not specified, so dry-run only")

    video_stem = osp.splitext(osp.basename(video_name))[0]
    dump_metric = dict(type='DumpResults',
                       out_file_path=osp.join(args.out_dir, 'extracted_features',
                                              f'{video_stem}_{args.modality}_feats.pkl'))
    cfg.test_evaluator = [dump_metric]

    # ---------------------- default settings ------------------------- #
    cfg.test_cfg = dict(type='TestLoop')
    cfg.default_scope = 'mmaction'

    cfg.default_hooks = dict(
        runtime_info=dict(type='RuntimeInfoHook'),
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=20, ignore_last=False),
        param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
        sampler_seed=dict(type='DistSamplerSeedHook'),
        sync_buffers=dict(type='SyncBuffersHook'))

    cfg.env_cfg = dict(
        cudnn_benchmark=False,
        mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
        dist_cfg=dict(backend='nccl'))

    cfg.log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)

    cfg.visualizer = dict(type='ActionVisualizer', vis_backends=[dict(type='LocalVisBackend')])

    cfg.log_level = 'INFO'
    cfg.resume = False

    return cfg


def main():
    # %% Parse configuration
    args = parse_args()
    if args.format == 'video':
        assert args.modality == 'RGB', "Extract optical flow features with raw video as input is not supported," \
                                       "You have to use the pre-extracted optical flow raw frames."
        # TODO: Add a on-line optical flow computer here to support this case.

    for video in Path(args.video_path).iterdir():
        cfg = get_cfg(args, video.name)
        runner = Runner.from_cfg(cfg)
        runner.logger.info(f"Processing video {video.name} ...")
        runner.test()


if __name__ == '__main__':
    main()
