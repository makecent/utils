# Config
```python
clip_len = 13
frame_interval = 6
num_clips = 1
dataset_type = 'RawframeDataset'
data_root = 'my_data/ucf101/rawframes/'
data_root_val = 'my_data/ucf101/rawframes/'
split = 1
ann_file_train = 'my_data/ucf101/ucf101_train_split_1_rawframes.txt'
ann_file_val = 'my_data/ucf101/ucf101_val_split_1_rawframes.txt'
ann_file_test = 'my_data/ucf101/ucf101_val_split_1_rawframes.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=13, frame_interval=6, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 182)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(160, 160), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=13,
        frame_interval=6,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 182)),
    dict(type='CenterCrop', crop_size=160),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=13,
        frame_interval=6,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 182)),
    dict(type='ThreeCrop', crop_size=182),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=24,
    workers_per_gpu=8,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RawframeDataset',
        start_index=0,
        ann_file='my_data/ucf101/ucf101_train_split_1_rawframes.txt',
        data_prefix='my_data/ucf101/rawframes/',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=13,
                frame_interval=6,
                num_clips=1),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 182)),
            dict(type='RandomResizedCrop'),
            dict(type='Resize', scale=(160, 160), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    val=dict(
        type='RawframeDataset',
        start_index=0,
        ann_file='my_data/ucf101/ucf101_val_split_1_rawframes.txt',
        data_prefix='my_data/ucf101/rawframes/',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=13,
                frame_interval=6,
                num_clips=1,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 182)),
            dict(type='CenterCrop', crop_size=160),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]),
    test=dict(
        type='RawframeDataset',
        start_index=0,
        ann_file='my_data/ucf101/ucf101_val_split_1_rawframes.txt',
        data_prefix='my_data/ucf101/rawframes/',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=13,
                frame_interval=6,
                num_clips=10,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 182)),
            dict(type='ThreeCrop', crop_size=182),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]))
optimizer = dict(
    type='AdamW',
    lr=0.001,
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_ratio=0.1,
    warmup_iters=2.5,
    warmup_by_epoch=True)
total_epochs = 30
checkpoint_config = dict(interval=5)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmaction/recognition/x3d/facebook/x3d_s_facebook_13x6x1_kinetics400_rgb_20201027-623825a0.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
custom_imports = dict(imports=['custom_modules'], allow_failed_imports=False)
model = dict(
    type='Recognizer3D',
    backbone=dict(type='X3D', gamma_w=1, gamma_b=2.25, gamma_d=2.2),
    cls_head=dict(type='X3DHead', num_classes=101, in_channels=432),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
work_dir = './work_dirs/x3d_s_ufc101_adamw30/'
gpu_ids = range(0, 2)
omnisource = False
module_hooks = []
```

# Val curve
![img.png](asserts/2.png)

# Test result
AdamW:
```shell
top1_acc: 0.9466                                                                                                                                                                                           
top5_acc: 0.9950    
```
SGD:
```shell
top1_acc: 0.8752
top5_acc: 0.9892
```
