custom_imports = dict(imports=['custom_modules'], allow_failed_imports=False)

# model settings
model = dict(
    type='APN',
    backbone=dict(type='MViT2', pretrained=False),
    cls_head=dict(
        type='APNHead',
        num_classes=400,
        in_channels=768,
        dropout_ratio=0.5,
        avg3d=False))
load_from = "/home/louis/PycharmProjects/APN/work_dirs/apn_mvit2_32x4_10e_kinetics400_rgb/epoch_10.pth"

# input configuration
clip_len = 32
frame_interval = 4

# dataset settings
dataset_type = 'DenseExtracting'
data_root = '/home/louis/PycharmProjects/APN/my_data/thumos14/rawframes/val'
ann_file = '/home/louis/PycharmProjects/APN/my_data/thumos14/annotations/apn/apn_val.csv'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
test_pipeline = [
    dict(type='FetchStackedFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs']),
]
data = dict(
    videos_per_gpu=1,
    workers_per_gpu=6,
    test=dict(
        type=dataset_type,
        ann_file=ann_file,
        data_prefix=data_root,
        pipeline=test_pipeline))

fp16 = dict()
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
dist_params = dict(backend='nccl')