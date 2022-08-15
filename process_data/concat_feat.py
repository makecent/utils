from pathlib import Path
from mmcv import track_iter_progress
import numpy as np

wd1 = Path("/home/louis/PycharmProjects/APN/my_data/thumos14/i3d_rgb_feature")
wd2 = Path("/home/louis/PycharmProjects/APN/my_data/thumos14/i3d_flow_feature")
to = Path("/home/louis/PycharmProjects/APN/my_data/thumos14/i3d_feature")
for i in track_iter_progress(list(wd1.iterdir())):
    f1 = np.load(str(i))
    f2 = np.load(str(wd2.joinpath(i.name)))
    f = np.hstack([f1.mean(axis=(-1, -2, -3)), f2.mean(axis=(-1, -2, -3))])
    np.save(str(to.joinpath(i.name)), f)
