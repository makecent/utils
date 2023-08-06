custom_imports = dict(imports=['custom_modules'], allow_failed_imports=False)
_base_ = ['../_base_/default_runtime.py']

# model settings
model = dict(
    type='FeatExtractor',
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[127.5, 127.5],
        std=[127.5, 127.5],
        format_shape='NCTHW'),
    backbone=dict(type='I3D',
                  modality='flow',
                  init_cfg=dict(type='Pretrained',
                                checkpoint='https://github.com/hassony2/kinetics_i3d_pytorch/raw/master/model/model_flow.pth')))
load_from = None
# input configuration
clip_len = 16   # the number of frames of each clip
frame_interval = 1  # the sampling interval between frames in a clip
clip_interval = 4   # the sampling interval between clips

test_pipeline = [
    dict(type='FetchStackedFrames', clip_len=clip_len, frame_interval=frame_interval, sampling_style='right'),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs']),
]
data = dict(
    videos_per_gpu=1,
    workers_per_gpu=6,
    test=dict(
        type='DenseExtracting',
        clip_interval=clip_interval,
        clip_len=clip_len * frame_interval,
        pipeline=test_pipeline,
        filename_tmpl='flow_{}_{:05}.jpg',
        modality='Flow'))