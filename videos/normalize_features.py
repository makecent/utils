# %% This python was created for normalizing the temporal length of features, specifically for ActionFormer features.
from pathlib import Path

import numpy as np
import torch.nn.functional


def normalize_feat():
    in_dim = 2048
    tgt_len = 100
    p = Path("/home/louis/PycharmProjects/loc_former/data/thumos/i3d_features")
    tgt_p = "/home/louis/PycharmProjects/loc_former/data/thumos/i3d_features_100/"
    for npy in p.iterdir():
        f = torch.from_numpy(np.load(str(npy)))
        f = torch.nn.functional.interpolate(f[None, None, :], size=[tgt_len, in_dim], mode='bilinear', align_corners=True)
        f = f.squeeze().numpy()
        np.save(tgt_p + npy.name, f)
