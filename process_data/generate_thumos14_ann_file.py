import pickle
from pathlib import Path

video_info = "my_data/thumos14/annotations/louis/video_info.pkl"
with open(video_info, 'rb') as f:
    video_info = pickle.load(f)


def generate_thumos14_ann_file(raw_ann):
    ann_dict = dict()
    for f in Path(raw_ann).iterdir():
        if f.stem == 'readme':
            continue
        cls_name = str(f.name).split('_')[0]
        with open(str(f), 'r') as file:
            for line in file.readlines():
                video_name, _, start_time, end_time = line.strip().rsplit(' ')
                if video_name == 'video_test_0001292':
                    # video_test_0001292 only has Ambiguous actions annotated, ignored
                    continue
                ann_dict.setdefault(video_name, {}).setdefault('segments', []).append(
                    [float(start_time), float(end_time)])
                ann_dict.setdefault(video_name, {}).setdefault('labels', []).append(cls_name)

                ann_dict.setdefault(video_name, {}).setdefault('duration', video_info[video_name]['dur'])
                ann_dict.setdefault(video_name, {}).setdefault('FPS', video_info[video_name]['fps'])
                ann_dict.setdefault(video_name, {}).setdefault('num_frame', video_info[video_name]['fra'])
                ann_dict.setdefault(video_name, {}).setdefault('segments_f', []).append(
                    [int(round(float(start_time) * float(video_info[video_name]['fps']))),
                     int(round(float(end_time) * float(video_info[video_name]['fps'])))])
    return ann_dict


def sort_ann_dict(ann_dict):
    # sort the annotation dictionary
    ann_dict = dict(sorted(ann_dict.items(), key=lambda x: x[0]))
    for k, v in ann_dict.items():
        segments, labels = v['segments'], v['labels']
        assert len(segments) == len(labels)
        sorted_inds = sorted(range(len(segments)), key=lambda i: (float(segments[i][0]),
                                                                  float(segments[i][1]),
                                                                  labels[i]))
        v['segments'] = [segments[i] for i in sorted_inds]
        v['labels'] = [labels[i] for i in sorted_inds]
    return ann_dict


ann_dict_val = sort_ann_dict(generate_thumos14_ann_file(raw_ann="my_data/thumos14/annotations/ori/annotations_val"))
ann_dict_test = sort_ann_dict(generate_thumos14_ann_file(raw_ann="my_data/thumos14/annotations/ori/annotations_test"))


def check_segments(ann_dict, dur_thr=0.1):
    too_short = {}
    out_of_range = {}
    for k, v in ann_dict.items():
        for seg in v['segments']:
            # check too short segments
            if (seg[1] - seg[0]) <= dur_thr:
                print(f"segment {seg} in {k} is too short (<= {dur_thr})")
                too_short.setdefault(k, []).append(seg)
            if not (0 <= seg[0] < seg[1] <= v['duration']):
                print(f"segment {seg} in {k} is out of range (duration={v['duration']})")
                out_of_range.setdefault(k, []).append(seg)
    return too_short, out_of_range


short1, out1 = check_segments(ann_dict_val, dur_thr=0.1)
# video_validation_0000364, the last two annotations are out of range (greater than video duration).
# video_validation_0000856, the last two annotations are out of range.
short2, out2 = check_segments(ann_dict_test, dur_thr=0.1)
# video_test_0000270, most annotations are out of range. After manual checking, the whole annotation may be wrong.
# video_test_0000814, the last three annotations are out of range.
# video_test_0001081, the last one annotation is out of range.

# By accident, video_test_0001496 was manually found to be wrong annotated.

print('end')
# video_validation_0000947 feature missed in feature.tar provided by TadTR

import json
with open('thumos14_val.json', 'w') as f:
    json.dump(ann_dict_val, f)
with open('thumos14_test.json', 'w') as f:
    json.dump(ann_dict_test, f)
