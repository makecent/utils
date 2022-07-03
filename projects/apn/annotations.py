# #%% ucf101 apn file generation
# import pandas as pd
# from pathlib import Path
# p = "data/ucf101/annotations/trainlist01.txt"
# new = "data/ucf101/annotations/apn_trainlist01.csv"
# frames = "data/ucf101/rawframes"
# action_ind = pd.read_csv("data/ucf101/annotations/classInd.txt", header=None, index_col=1, sep=' ').T
#
# with open(p, 'r') as f:
#     with open(new, 'w') as n:
#         for line in f:
#             folder = line.split('.')[0]
#             action = folder.split('/')[0]
#             ind = action_ind[action].values[0] - 1
#             raw_frames = Path(frames, folder)
#             total_frame = len(list(raw_frames.glob("flow_y_*.jpg")))
#             n.write(f"{folder},{total_frame},0,{total_frame-1},{ind}\n")

# #%% activitynet1.3 (5fps) apn file generation
# import json
# import numpy as np
# raw_frame_path = 'data/ActivityNet/rawframes_5fps'
# with open("data/ActivityNet/annotations/activity_1_3.json", 'r') as f:
#     ann = json.load(f)['database']
#
# with open("data/ActivityNet/annotations/aty_classind.json", 'r') as f2:
#     classind = json.load(f2)
#
# with open("data/ActivityNet/apn_aty_train_5fps.csv", 'w') as n:
#     for video_id, video_info in ann.items():
#         if video_info['subset'] == 'training':
#             total_frames = len(list(Path(raw_frame_path, 'v_'+video_id).iterdir()))
#             for seg in video_info['annotations']:
#                 start, end = np.array(seg['segment'])/video_info['duration']*total_frames
#                 indlabel = classind[seg['label']]
#                 n.write(f"{video_id},{total_frames},{round(start)},{round(end)},{indlabel}\n")
#
# with open("data/ActivityNet/apn_aty_test_5fps.csv", 'w') as n:
#     for video_id, video_info in ann.items():
#         if video_info['subset'] == 'testing':
#             total_frames = len(list(Path(raw_frame_path, 'v_'+video_id).iterdir()))
#             start, end = 0, 0
#             indlabel = 0
#             n.write(f"{video_id},{total_frames},{start},{end},{indlabel}\n")

#%% activitynet1.3 apn file generation
import json
import numpy as np
from pathlib import Path
from mmaction.datasets.pipelines import DecordInit
from os import path as osp
from mmcv import dump, track_iter_progress

video_reader = DecordInit()
video_path = '/home/louis/PycharmProjects/ProgPreTrain/my_data/activitynet/videos/val_resized'
with open("/home/louis/PycharmProjects/ProgPreTrain/my_data/activitynet/annotations/original/activity_net.v1-3.min.json", 'r') as f:
    ann = json.load(f)['database']

# mapping string label to int label
class_label = []
for video_id, video_info in ann.items():
    for seg in video_info['annotations']:
        label = seg['label']
        class_label.append(label)
class_label = sorted(np.unique(class_label))
class_dict = dict()
for i, l in enumerate(class_label):
    class_dict[l] = i
# dump(class_dict, '/home/louis/PycharmProjects/ProgPreTrain/my_data/activitynet/annotations/apn/aty_label_mapping.json')

# with open("/home/louis/PycharmProjects/ProgPreTrain/my_data/activitynet/annotations/apn/apn_aty_train_video.csv", 'w') as n:
#     for video_id, video_info in track_iter_progress(ann.items()):
#         video_name = 'v_' + video_id + '.mp4'
#         if video_info['subset'] == 'training':
#             total_frames = video_reader(dict(filename=f'{osp.join(video_path, video_name)}'))['total_frames']
#             for seg in video_info['annotations']:
#                 start, end = np.array(seg['segment'])
#                 if start == 0.01 and end == 0.02:
#                     continue
#                 if start == 0 and end == 0:
#                     continue
#                 if start > video_info["duration"]:
#                     continue
#                 if end - start < 1:
#                     # print(f'video {video_name} has segment {seg["segment"]} less than 1 second. Abandoned')
#                     continue
#                 if end > video_info["duration"]*1.01:
#                     # print(f'video {video_name} has annotations {seg["segment"]} out of range {video_info["duration"]}')
#                     continue
#
#                 start *= total_frames/video_info['duration']
#                 end *= total_frames/video_info['duration']
#                 start = min(start, total_frames)
#                 end = min(end, total_frames)
#
#                 indlabel = class_dict[seg['label']]
#                 n.write(f"{video_name},{total_frames},{round(start)},{round(end)},{indlabel}\n")

with open("/home/louis/PycharmProjects/ProgPreTrain/my_data/activitynet/annotations/apn/apn_aty_val_video.csv", 'w') as n:
    for video_id, video_info in track_iter_progress(ann.items()):
        video_name = 'v_' + video_id + '.mp4'
        if video_info['subset'] == 'validation':
            total_frames = video_reader(dict(filename=f'{osp.join(video_path, video_name)}'))['total_frames']
            for seg in video_info['annotations']:
                start, end = np.array(seg['segment'])
                if start == 0.01 and end == 0.02:
                    continue
                if start == 0 and end == 0:
                    continue
                if start > video_info["duration"]:
                    continue
                if end > video_info["duration"]*1.01:
                    print(f'video {video_name} has annotations {seg["segment"]} out of range {video_info["duration"]}')
                    continue

                start *= total_frames/video_info['duration']
                end *= total_frames/video_info['duration']
                start = min(start, total_frames)
                end = min(end, total_frames)

                indlabel = class_dict[seg['label']]
                n.write(f"{video_name},{total_frames},{round(start)},{round(end)},{indlabel}\n")


# # %% thumos14 apn annotations generation
# CLASSES = ('BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
#            'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving',
#            'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump',
#            'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
#            'SoccerPenalty', 'TennisSwing', 'ThrowDiscus',
#            'VolleyballSpiking')
# import mmcv
# import pandas as pd
# from pathlib import Path
#
#
# def get_video_length(video_path):
#     import cv2
#     v = cv2.VideoCapture(video_path)
#     fps = v.get(cv2.CAP_PROP_FPS)
#     num_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
#     duration = round(num_frames / fps, 2)
#     return duration, num_frames
#
#
# def generate_th14_ann(video_directory, rawframe_directory, ann_ori, out):
#     """Example
#     video_directory = f'/home/louis/PycharmProjects/APN/data/thumos14/videos/val/'
#     rawframe_directory = f'/home/louis/PycharmProjects/APN/data/thumos14/rawframes/val/'
#     ann_ori = '/home/louis/PycharmProjects/FCOS-TAL/data/thumos14/annotations_val'
#     """
#     segments = []
#     video_info = {}
#     for ann_file in mmcv.track_iter_progress(list(Path(ann_ori).iterdir())):
#         if 'Ambiguous' in ann_file.name or 'readme' in ann_file.name:
#             continue
#         label = ann_file.stem.split('_')[0]
#         ann = pd.read_csv(ann_file, header=None, sep=' ')
#         for i, row in ann.iterrows():
#             seg_info = {}
#             video_name = row[0]
#             # skip two wrong annotated videos
#             if video_name == 'video_test_0000270' or video_name == 'video_test_0001496':
#                 continue
#             start, end = row[2:4]
#
#             video_frame_num = video_info.get(video_name, {}).get(
#                 'video_frame_num', 0)
#             video_duration = video_info.get(video_name, {}).get(
#                 'video_duration', 0)
#
#             if video_frame_num == 0:
#                 video_path = video_directory + video_name + '.mp4'
#                 video_duration, video_frame_num = get_video_length(video_path)
#                 rawframes_num = len(
#                     list(Path(rawframe_directory + video_name).glob('img*')))
#                 if rawframes_num != video_frame_num:
#                     print(
#                         f'\nnumber of raw frames extracted from {video_name} does not meet the information got from the video file')
#                 video_info.setdefault(video_name, {}).setdefault(
#                     'video_frame_num', video_frame_num)
#                 video_info.setdefault(video_name, {}).setdefault(
#                     'video_duration', video_duration)
#
#             if start > video_duration or end > video_duration:
#                 # Find wrong segments: row 190, 191 in HighJump_val.txt; row 47, 48 in SoccerPenalty_val.txt;
#                 # row 320 in Diving_test.txt; and row 34, 35, 36 in ThrowDiscus_test.txt.
#                 print(
#                     f'\nsegment {[start, end]} in {video_name} out of range {video_duration} ({ann_file.name}: row {i + 1})')
#                 continue
#
#             seg_info['video_name'] = video_name
#             # minus 1 to align with optical flows
#             seg_info['total_frames'] = video_frame_num - 1
#             seg_info['segment'] = [
#                 int(round(video_frame_num * start / video_duration)),
#                 int(round(video_frame_num * end / video_duration))]
#             seg_info['label'] = CLASSES.index(label)
#             segments.append(seg_info)
#
#     with open(out, 'w') as f:
#         for s in segments:
#             f.writelines(
#                 f"{s['video_name'] + '.mp4'},{s['total_frames']},{s['segment'][0]},{s['segment'][1]},{s['label']}\n")
#
#     return None
#
#
# # t1 = generate_th14_ann(
# #     video_directory=f'/home/louis/PycharmProjects/APN/data/thumos14/videos/val/',
# #     rawframe_directory=f'/home/louis/PycharmProjects/APN/data/thumos14/rawframes/val/',
# #     ann_ori='/home/louis/PycharmProjects/FCOS-TAL/data/thumos14/annotations/ori/annotations_val',
# #     out='apn_val.csv',
# # )
# # t2 = generate_th14_ann(
# #     video_directory=f'/home/louis/PycharmProjects/APN/data/thumos14/videos/test/',
# #     rawframe_directory=f'/home/louis/PycharmProjects/APN/data/thumos14/rawframes/test/',
# #     ann_ori='/home/louis/PycharmProjects/FCOS-TAL/data/thumos14/annotations/ori/annotations_test',
# #     out='apn_test.csv',
# # )
#
# train_frame_path = "/home/louis/PycharmProjects/APN/data/thumos14/rawframes/train/"
# train_video_path = "/home/louis/PycharmProjects/APN/data/thumos14/videos/train"
#
# with open('apn_train_video.csv', 'w') as f:
#     for v in sorted(list(Path(train_video_path).iterdir())):
#         label = CLASSES.index(v.stem.split('_')[1])
#         _, videoframes_num = get_video_length(str(v))
#         # minus 1 to align with optical flows
#         videoframes_num += -1
#         f.writelines(f"{v.name},{videoframes_num},{0},{videoframes_num-1},{label}\n")
#
# with open('apn_train_frame.csv', 'w') as f:
#     for v in sorted(list(Path(train_frame_path).iterdir())):
#         label = CLASSES.index(v.stem.split('_')[1])
#         rawframes_num = len(list(v.glob('img*')))
#         # minus 1 to align with optical flows
#         rawframes_num += -1
#         f.writelines(f"{v.name+'.avi'},{rawframes_num},{0},{rawframes_num-1},{label}\n")

