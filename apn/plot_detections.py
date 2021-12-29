#%% plot quantitive results
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# detections = 'apn/src/th14_detections.pkl'
detections = 'apn/src/dfmad_detections.pkl'
gts = 'apn/src/apn_dfmad_test.csv'
with open(detections, 'rb') as f:
    dets = pickle.load(f)
gts = pd.read_csv(gts, header=None)
gts[0] = gts[0].astype('string').apply(lambda x: '.'.join(x.rsplit('.')[0]))
video_names = np.unique(gts[0].values)

video_idx = 1
video_name = (video_names[video_idx])

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
color = {0: 'r', 1: 'g', 2: 'y'}
for cls_idx in det.keys():
    det_of_cls = det[cls_idx]
    gt_of_cls = gt[cls_idx]
    for i, g in enumerate(gt_of_cls):
        start, end = g
        label = f'ac{cls_idx}' if i ==0 else None
        t1 = plt.bar((end+start)/2, width=end-start, height=10, color=color[cls_idx], alpha=1, edgecolor='k', linewidth=1, label=label)
    for i, d in enumerate(det_of_cls):
        start, end, score = d
        if cls_idx == 0 and i >= 4:
            break
        if cls_idx == 1 and i >= 2:
            break
        if cls_idx == 2 and i >= 2:
            break
        plt.bar((end+start)/2, width=end-start, height=10, bottom=15, color=color[cls_idx], alpha=1, edgecolor='k',)
# plt.legend()
plt.show()