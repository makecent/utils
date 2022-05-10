from mmcv.parallel import MMDataParallel
from mmaction.models import build_model
from mmaction.datasets import build_dataset, build_dataloader
from torch.nn.functional import cross_entropy
from fvcore.nn import parameter_count
import torch
import mmcv

def single_gpu_test(model, data_loader, mmlab=True):
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            if mmlab:
                result = model(return_loss=False, **data)
            else:
                result = model(data['imgs'].squeeze(1))

        # Assume result has the same length of batch_size
        # refer to https://github.com/open-mmlab/mmcv/issues/985
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()


def single_gpu_train(model, data_loader, mmlab=True):
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        if mmlab:
            loss = model(**data)['loss_cls']
            loss.backward()
        else:
            result = model(data['imgs'].squeeze(1))
            loss = cross_entropy(result, data['label'].to(result.device))
            loss.backward()

        # Assume result has the same length of batch_size
        # refer to https://github.com/open-mmlab/mmcv/issues/985
        batch_size = len(data['imgs'])
        for _ in range(batch_size):
            prog_bar.update()


# model = torch.hub.load("facebookresearch/pytorchvideo", model='x3d_m', pretrained=False)
model = build_model(
    dict(
        type='Recognizer3D',
        backbone=dict(type='X3D', gamma_w=1, gamma_b=2.25, gamma_d=2.2),
        cls_head=dict(
            type='X3DHead',
            in_channels=432,
            num_classes=400,
            spatial_type='avg',
            dropout_ratio=0.5,
            fc1_bias=False),
        # model training and testing settings
        train_cfg=None,
        test_cfg=dict(average_clips='prob'))
)
# model = build_model(
#     dict(
#         type='Recognizer3D',
#         backbone=dict(
#             type='ResNet3d',
#             pretrained2d=True,
#             pretrained='torchvision://resnet50',
#             depth=50,
#             conv1_kernel=(5, 7, 7),
#             conv1_stride_t=2,
#             pool1_stride_t=2,
#             conv_cfg=dict(type='Conv3d'),
#             norm_eval=False,
#             inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
#             zero_init_residual=False),
#         cls_head=dict(
#             type='I3DHead',
#             num_classes=400,
#             in_channels=2048,
#             spatial_type='avg',
#             dropout_ratio=0.5,
#             init_std=0.01),
#         # model training and testing settings
#         train_cfg=None,
#         test_cfg=dict(average_clips='prob'))
# )
# model = build_model(
#     dict(
#     type='Recognizer3D',
#     backbone=dict(
#         type='ResNet3d_sony',
#         modality='rgb'),
#     cls_head=dict(
#         type='I3DHead',
#         num_classes=400,
#         in_channels=1024,
#         spatial_type='avg',
#         dropout_ratio=0.5,
#         init_std=0.01),
#     # model training and testing settings
#     train_cfg=None,
#     test_cfg=dict(average_clips='prob')))
model = MMDataParallel(model, device_ids=[0])

# dataset settings
dataset_type = 'VideoDataset'
data_root = 'my_data/kinetics400/videos_train'
data_root_val = 'my_data/kinetics400/videos_val'
ann_file_train = 'my_data/kinetics400/kinetics400_train_list_videos.txt'
ann_file_val = 'my_data/kinetics400/kinetics400_val_list_videos.txt'
ann_file_test = 'my_data/kinetics400/kinetics400_val_list_videos.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
val_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
dataset = build_dataset(dict(
    type=dataset_type,
    ann_file=ann_file_val,
    data_prefix=data_root_val,
    pipeline=val_pipeline),
    dict(test_mode=True))

data_loader = build_dataloader(dataset,
                               videos_per_gpu=64,  # 4 for train
                               workers_per_gpu=2,
                               dist=False,
                               shuffle=False)
# print(parameter_count(model)[''])
# single_gpu_test(x3d_ori, data_loader, mmlab=False)
single_gpu_test(model, data_loader)

# single_gpu_train(model, data_loader, mmlab=False)
# single_gpu_train(model, data_loader)
