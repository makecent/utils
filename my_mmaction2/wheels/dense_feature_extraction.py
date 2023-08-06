# To use it, first add `utils` to PythonPath:
# `$env:PYTHONPATH += ";$pwd"` for Windows, `PYTHONPATH=$PWD:$PYTHONPATH` for Linux
# then run
# python my_mmaction2/wheels/dense_feature_extraction.py
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
    parser.add_argument('--format', choices=['Video', 'RawFrames'], default='Video',
                        help='format of video files in video path, raw videos or pre-decoded raw frames')
    parser.add_argument('--modality', choices=['RGB', 'Flow'], default='RGB',
                        help='the format of feature to be extracted')
    parser.add_argument('--filename-tmpl', default='img_{:05}.jpg',
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
    parser.add_argument('--local_rank', type=int, default=0)
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
        backbone=dict(type='I3D',
                      modality=f'{args.modality.lower()}',
                      init_cfg=dict(type='Pretrained',
                                    checkpoint=f'https://github.com/hassony2/kinetics_i3d_pytorch/raw/master/model/model_{args.modality.lower()}.pth')))

    # ----------------------- data settings ------------------------- #
    if args.format == 'Video':
        pipeline = [dict(type='DecordInit'), dict(type='DecordDecode')]
    else:
        pipeline = [dict(type='RawFrameDecode')]
    pipeline.extend([dict(type='Resize', scale=(-1, 256)),
                     dict(type='CenterCrop', crop_size=224),
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
            vformat=args.format))

    # -------------------- Dump predictions --------------------
    if args.out_dir is not None:
        mkdir_or_exist(args.out_dir)
        cfg.work_dir = osp.join(args.out_dir, 'extracted_features')
    else:
        cfg.work_dir = 'work_dir'
        warnings.warn("Output directory was not specified, so dry-run only")

    dump_metric = dict(type='DumpResults',
                       out_file_path=osp.join(cfg.work_dir, f'{video_name}_{args.modality}_feats.pkl'))
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
    cfg.load_from = None
    cfg.resume = False

    return cfg


def main():
    # %% Parse configuration
    args = parse_args()
    if args.format == 'Video':
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
