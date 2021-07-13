#%% ucf101 apn file generation
import pandas as pd
from pathlib import Path
p = "data/ucf101/annotations/trainlist01.txt"
new = "data/ucf101/annotations/apn_trainlist01.csv"
frames = "data/ucf101/rawframes"
action_ind = pd.read_csv("data/ucf101/annotations/classInd.txt", header=None, index_col=1, sep=' ').T

with open(p, 'r') as f:
    with open(new, 'w') as n:
        for line in f:
            folder = line.split('.')[0]
            action = folder.split('/')[0]
            ind = action_ind[action].values[0] - 1
            raw_frames = Path(frames, folder)
            total_frame = len(list(raw_frames.glob("flow_y_*.jpg")))
            n.write(f"{folder},{total_frame},0,{total_frame-1},{ind}\n")

#%% activitynet1.3 apn file generation
import json
import numpy as np
raw_frame_path = 'data/ActivityNet/rawframes_5fps'
with open("data/ActivityNet/annotations/activity_1_3.json", 'r') as f:
    ann = json.load(f)['database']

with open("data/ActivityNet/annotations/aty_classind.json", 'r') as f2:
    classind = json.load(f2)

with open("data/ActivityNet/apn_aty_train_5fps.csv", 'w') as n:
    for video_id, video_info in ann.items():
        if video_info['subset'] == 'training':
            total_frames = len(list(Path(raw_frame_path, 'v_'+video_id).iterdir()))
            for seg in video_info['annotations']:
                start, end = np.array(seg['segment'])/video_info['duration']*total_frames
                indlabel = classind[seg['label']]
                n.write(f"{video_id},{total_frames},{round(start)},{round(end)},{indlabel}\n")

with open("data/ActivityNet/apn_aty_test_5fps.csv", 'w') as n:
    for video_id, video_info in ann.items():
        if video_info['subset'] == 'testing':
            total_frames = len(list(Path(raw_frame_path, 'v_'+video_id).iterdir()))
            start, end = 0, 0
            indlabel = 0
            n.write(f"{video_id},{total_frames},{start},{end},{indlabel}\n")