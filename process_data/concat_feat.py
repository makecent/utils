# Concatenate rgb and flow feature
from pathlib import Path
from mmengine.utils import track_iter_progress
import numpy as np

rgb = Path("/home/louis/PycharmProjects/TAD_DINO/my_data/thumos14/features/thumos_feat_VideoMAE2_16input_4stride_1408_RGB")
flow = Path("/home/louis/PycharmProjects/TAD_DINO/my_data/thumos14/features/thumos_feat_ActionFormer_16input_4stride_2048/i3d_features")
to = Path("/home/louis/PycharmProjects/TAD_DINO/my_data/thumos14/features/thumos_feat_VideoMAE2-RGB_I3D-RGB")
for i in track_iter_progress(list(rgb.iterdir())):
    rgb_feat = np.load(str(i))
    flow_feat = np.load(str(flow.joinpath(i.name)))[:, :1024]
    if rgb_feat.shape[0] != flow_feat.shape[0]:
        assert rgb_feat.shape[0] == flow_feat.shape[0] + 1
        rgb_feat = rgb_feat[:-1]
    f = np.hstack([rgb_feat, flow_feat])
    np.save(str(to.joinpath(i.name)), f)
