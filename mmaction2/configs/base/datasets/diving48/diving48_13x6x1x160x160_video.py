clip_len = 13
frame_interval = 6
num_clips = 1

# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/diving48/videos'
data_root_val = 'data/diving48/videos'
ann_file_train = 'data/diving48/diving48_train_list_videos.txt'
ann_file_val = 'data/diving48/diving48_val_list_videos.txt'
ann_file_test = 'data/diving48/diving48_val_list_videos.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval, num_clips=num_clips),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 182)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(160, 160), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval, num_clips=num_clips, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 182)),
    dict(type='CenterCrop', crop_size=160),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval, num_clips=num_clips, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 182)),
    dict(type='ThreeCrop', crop_size=160),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=8,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
