import mmengine


def generate_ant_ann_file(ant_annfile, ant_video_info):
    ant_annfile = mmengine.load(ant_annfile)
    ant_val_dict = dict()
    ant_train_dict = dict()
    for video_name, ann in ant_annfile['database'].items():
        split = ann['subset']
        ann_dict = ant_train_dict if split == 'training' else ant_val_dict
        ann_dict.setdefault(video_name, {}).setdefault('duration', ann['duration'])
        ann_dict.setdefault(video_name, {}).setdefault('FPS', ant_video_info[video_name]['fps'])
        for seg in ann['annotations']:
            se, cls_name = seg['segment'], seg['label']
            start_time, end_time = se
            ann_dict.setdefault(video_name, {}).setdefault('segments', []).append(
                [float(start_time), float(end_time)])
            ann_dict.setdefault(video_name, {}).setdefault('labels', []).append(cls_name)

            ann_dict.setdefault(video_name, {}).setdefault('duration', ant_video_info[video_name]['duration'])
            ann_dict.setdefault(video_name, {}).setdefault('FPS', ant_video_info[video_name]['fps'])
            ann_dict.setdefault(video_name, {}).setdefault('num_frame', ant_video_info[video_name]['num_frame'])
            ann_dict.setdefault(video_name, {}).setdefault('segments_f', []).append(
                [int(round(float(start_time) * float(ant_video_info[video_name]['fps']))),
                 int(round(float(end_time) * float(ant_video_info[video_name]['fps'])))])
    return ant_train_dict, ant_val_dict


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


ann_train_dict, ann_val_dict = sort_ann_dict(generate_ant_ann_file(
    ant_annfile=r"C:\Users\lucho\PycharmProjects\TAD_DINO\my_data\ant\annotations\anet1.3_i3d_filtered.json",
    ant_video_info=r'ant_video_info.json'))


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


short1, out1 = check_segments(ann_train_dict, dur_thr=0.1)
short2, out2 = check_segments(ann_val_dict, dur_thr=0.1)
print('end')

import json

with open('thumos14_val.json', 'w') as f:
    json.dump(ann_train_dict, f)
with open('thumos14_test.json', 'w') as f:
    json.dump(ann_val_dict, f)
