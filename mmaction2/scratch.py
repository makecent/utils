from mmcv import Config
from mmaction.datasets import build_dataset
import numpy as np

cfg = Config.fromfile("configs/apn/feature_extraction/apn_mvit2_32x4_k400_thumos14_rgb.py")
ann_file = [x.strip() for x in open(cfg.data.test.ann_file).readlines()]
for ann_line in ann_file:
    cfg.data.test.ann_file = ann_line
    video_name = ann_line.rsplit('.', 1)[0]
    ds = build_dataset(cfg.data.test)
    print(f"Len of extracted feat  of video {video_name} \t\t = {len(ds)}")
    feat = np.load(f"/home/louis/PycharmProjects/actionformer_release/data/thumos/i3d_features/{video_name}.npy")
    print(f"Len of actionformer feat of video {video_name} \t = {feat.shape[0]}")
    if len(ds) != feat.shape[0]:
        raise ValueError

print('haha')