# This function is revised based on the mmaction2.data.activitynet.tsn_feature_extraction.py
import argparse
import os
import os.path as osp
import mmcv
import pandas as pd
import torch

from mmaction.datasets.pipelines import Compose
from mmaction.models import build_backbone
from i3d_model import I3D


def parse_args():
    parser = argparse.ArgumentParser(description='Extract I3D Feature')
    parser.add_argument('--data-prefix', default='', help='dataset prefix')
    parser.add_argument('--output-prefix', default='', help='output prefix')
    parser.add_argument(
        '--data-list',
        help='video list of the dataset, the format should be '
             '`frame_dir num_frames class_label(anything else is ok)`')
    parser.add_argument(
        '--segment-length',
        type=int,
        default=64,
        help='the number of frame in each segments for dividing video')
    parser.add_argument('--modality', default='RGB', choices=['RGB', 'Flow'])
    parser.add_argument('--batch-size', type=int, default=2, help='the number of segments handle in each pass')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.is_rgb = args.modality == 'RGB'
    pretrained = "https://github.com/hassony2/kinetics_i3d_pytorch/raw/master/model/model_{modality}.pth".format(modality='rgb' if args.is_rgb else 'flow')
    args.clip_len = 1 if args.is_rgb else 5
    rgb_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False)
    flow_norm_cfg = dict(mean=[128, 128], std=[128, 128])
    args.img_norm_cfg = rgb_norm_cfg if args.is_rgb else flow_norm_cfg
    args.f_tmpl = 'img_{:05d}.jpg' if args.is_rgb else 'flow_{}_{:05d}.jpg'
    args.in_channels = args.clip_len * (3 if args.is_rgb else 2)

    # define the data pipeline for Untrimmed Videos
    data_pipeline = [
        dict(
            type='UntrimmedSampleFrames',
            clip_len=args.segment_length,
            frame_interval=args.segment_length),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(224, 224), keep_ratio=False),
        dict(type='Normalize', **args.img_norm_cfg),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='Collect', keys=['imgs'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    data_pipeline = Compose(data_pipeline)

    # define TSN R50 model, the model is used as the feature extractor
    backbone_cfg = dict(
            type='I3D',
            init_cfg=dict(type='Pretrained', checkpoint=pretrained),
            modality='rgb' if args.is_rgb else 'flow')
    model = build_backbone(backbone_cfg)
    model.init_weights()  # weights in cls_head, i.e. conv3d_0c_1x1., won't be used.
    model = model.cuda()
    model.eval()

    data = pd.read_csv(args.data_list, header=None, delimiter=' ').drop_duplicates(subset=0).values.tolist()

    # enumerate Untrimmed videos, extract feature from each of them
    prog_bar = mmcv.ProgressBar(len(data))
    if not osp.exists(args.output_prefix):
        os.system(f'mkdir -p {args.output_prefix}')

    for item in data:
        frame_dir, length, _ = item
        frame_dir = str(frame_dir)
        output_file = osp.basename(frame_dir)
        frame_dir = osp.join(args.data_prefix, frame_dir)
        output_file = osp.join(args.output_prefix, output_file)
        length = int(length)

        # prepare a pseudo sample
        tmpl = dict(
            frame_dir=frame_dir,
            total_frames=length,
            start_index=0,
            filename_tmpl=args.f_tmpl,
            modality=args.modality)
        sample = data_pipeline(tmpl)
        imgs = sample['imgs']

        def forward_data(model, data):
            # chop large data into pieces and extract feature from them
            results = []
            start_idx = 0
            num_segs = data.shape[0]
            while start_idx < num_segs:
                with torch.no_grad():
                    seg = data[start_idx:start_idx + args.batch_size]
                    feat = model.forward(seg.cuda())
                    feat = feat.flatten(start_dim=1)
                    results.append(feat.cpu())
                    start_idx += args.batch_size
            return torch.concat(results)

        feat = forward_data(model, imgs)
        torch.save(feat, output_file)
        prog_bar.update()


if __name__ == '__main__':
    main()
