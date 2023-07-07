custom_imports = dict(imports=['custom_modules'], allow_failed_imports=False)
# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(type='I3D',
                  init_cfg=dict(type='Pretrained',
                                checkpoint='https://github.com/hassony2/kinetics_i3d_pytorch/raw/master/model/model_flow.pth'),
                  modality='flow'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'))

# dataset settings
dataset_type = 'RawframeDataset'
data_root_val = 'my_data/thumos14/rawframes/val'
ann_file_val = 'my_data/thumos14/annotations/mmaction/thumos14_val_flow_list.txt'


test_pipeline = [
    dict(type='UntrimmedSampleFrames', clip_len=16, frame_interval=1, clip_interval=4),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='CenterCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

# default settings
test_cfg = dict(type='TestLoop')
default_scope = 'mmaction'

default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

log_level = 'INFO'
load_from = None
resume = False
# PYTHONPATH='.':$PYTHONPATH mim run mmaction2 clip_feature_extraction
# configs/feature_extraction/new_i3d_16x1x4_k400_thumos14_flow.py checkpoints/r3d_sony/model_flow_backbone-prefix.pth flow_clip_feat --long-video-mode