#%% plot quantitive results
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def detbycls_to_detbyvid(dets):
    new_dets = {}
    for cls, value in dets.items():
        for d in value:
            vid_name, _, start, end, score = d
            new_dets.setdefault(vid_name, {}).setdefault(int(cls), []).append([float(start), float(end), float(score)])
    for k, v in new_dets.items():
        for k2, v2 in v.items():
            new_dets[k][k2] = np.array(v2)
    return new_dets


# detections = 'apn/src/th14_detections.pkl'
# gts = 'apn/src/apn_th14_test.csv'
detections = 'apn/src/dfmad_detections.pkl'
gts = 'apn/src/apn_dfmad_test.csv'
with open(detections, 'rb') as f:
    dets = pickle.load(f)
if isinstance(list(dets.keys())[0], int):
    dets = detbycls_to_detbyvid(dets)

gts = pd.read_csv(gts, header=None)
gts[0] = gts[0].astype('string').apply(lambda x: x.rsplit('.')[0])
video_names = np.unique(gts[0].values)

video_idx = 1
video_name = (video_names[video_idx])
# video_name = 'video_test_0000882'

det = dets[video_name]
gt_x = gts[gts[0] == video_name]
gt = {}
for cls_idx in det.keys():
    gt[cls_idx] = gt_x[gt_x[4] == cls_idx].iloc[:, 2: 4].values

num_frames = gt_x[1].iloc[0]


axes = plt.gca()
fig = plt.gcf()
fig.set_size_inches(7*2.54, 2)
plt.xlim(-num_frames*.15, num_frames*1.15)
plt.ylim(-2, 32)
plt.bar(num_frames/2, width=num_frames, height=10, fill=False, edgecolor='k', linewidth=1)
plt.bar(num_frames/2, width=num_frames, height=10, bottom=15, fill=False, edgecolor='k', linewidth=1)
axes.get_yaxis().set_visible(False)
axes.get_xaxis().set_visible(False)
plt.box(False)
color_dict = {0: 'r', 1: 'g', 2: 'y'}
for cls_idx in det.keys():
    color = color_dict[list(det.keys()).index(cls_idx)]
    det_of_cls = det[cls_idx]
    gt_of_cls = gt[cls_idx]
    for i, g in enumerate(gt_of_cls):
        start, end = g
        label = f'ac{cls_idx}' if i == 0 else None
        t1 = plt.bar((end+start)/2, width=end-start, height=10, bottom=0, color=color, alpha=1, edgecolor='k', linewidth=1, label=label)
    for j, d in enumerate(det_of_cls):
        start, end, score = d
        if j > i:
            break
        plt.bar((end+start)/2, width=end-start, height=10, bottom=15, color=color, alpha=1, edgecolor='k',)
# plt.legend()
plt.show()