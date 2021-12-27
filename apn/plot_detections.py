#%% plot quantitive results
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
th14_detections = 'work_dirs/apn_coral+random_r3dsony_32x4_10e_thumos14_flow/mean_progressions.pkl'
dfmad_detections = 'work_dirs/apn_coral+random_r3dsony_32x4_10e_dfmad_rgb/mean_fuse/detections.pkl'
dfmad_gts = 'my_data/dfmad70/apn_test.csv'
with open(dfmad_detections, 'rb') as f:
    dets = pickle.load(f)
gts = pd.read_csv(dfmad_gts, header=None)
gts[0] = gts[0].astype('string')
video_names = np.unique(gts[0].values)

video_idx = 1
video_name = (video_names[video_idx])
det = dets[video_name]
gt_x = gts[gts[0] == video_name]
gt = {}
for cls_idx in det.keys():
    gt[cls_idx] = gt_x[gt_x[4] == cls_idx].iloc[:, 2: 4].values
    gt['num_frames'] = gt_x[1].iloc[0]

plt.xlim(0, gt['num_frames'])
color = {0: 'r', 1: 'g', 2: 'y'}
for cls_idx in det.keys():
    det_of_cls = det[cls_idx]
    gt_of_cls = gt[cls_idx]
    for g in gt_of_cls:
        start, end = g
        plt.bar((end+start)/2, width=end-start, height=120, color=color[cls_idx], alpha=0.5, edgecolor='k', linewidth=1)
    for i, d in enumerate(det_of_cls):
        start, end, score = d
        if cls_idx == 0 and i >= 4:
            break
        if cls_idx == 1 and i >= 2:
            break
        if cls_idx == 2 and i >= 2:
            break
        plt.bar((end+start)/2, width=end-start, height=100, color=color[cls_idx], alpha=0.4)
plt.show()