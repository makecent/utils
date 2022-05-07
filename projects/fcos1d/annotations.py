#%% generate annotations
import mmcv
from pathlib import Path


def get_frames_num(video_path):
    import cv2
    num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    return num_frames

video_directory = f'/home/louis/PycharmProjects/APN/data/ActivityNet/videos/val_resized'
ann_ori = '/home/louis/PycharmProjects/FCOS-TAL/data/ActivityNet/annotations/ori/activity_net.v1-3.min.json'
anno_database = mmcv.load(ann_ori)['database']

new_ann = []
skipped = 0
for video_name, data in mmcv.track_iter_progress(anno_database.items()):
    if data['subset'] != 'validation':
        continue
    new_video_name = f"v_{video_name}"
    video_path = list(Path(video_directory).glob(f"*{video_name}*"))[0]
    # frame_num = get_frames_num(str(video_path))
    # assert frame_num != 0
    gt_bboxes = []
    gt_labels = []
    for seg in data['annotations']:
        # seg_by_frame = [int(round(frame_num * i/data['duration'])) for i in seg['segment']]
        # if seg_by_frame[1] - seg_by_frame[0] <= 3:
        #     print(f"\n segment {seg} in video {video_name} has less than 3 frames ({seg_by_frame[1] - seg_by_frame[0]})")
        #     continue
        segment = seg['segment']
        if segment[1] - segment[0] <= 0.1:
            print(f"\n segment {seg} in video {video_name} last less than 0.1 seconds ({segment[0], segment[1]}), skipped")
            skipped += 1
            continue
        gt_bboxes.append(seg['segment'])
        gt_labels.append(seg['label'])
    if gt_labels:
        new_ann.append({'filename': video_path.name,
                        'duration': data['duration'],
                        'gt_tboundaries': gt_bboxes,
                        'gt_labels': gt_labels})
mmcv.dump(new_ann, 'val.json')