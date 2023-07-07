# generate ann_file for thumos14
from pathlib import Path

raw_frame_path = Path('/home/louis/PycharmProjects/APN/my_data/thumos14/rawframes/val')
lines = []
for video_folder in sorted(raw_frame_path.iterdir()):
    num_frames = len(list(video_folder.glob('img*'))) - 1
    pseudo_label = 0
    lines.append(f'{video_folder.name} {num_frames} {pseudo_label}\n')  # video_name num_frames pseudo_label

with open('/home/louis/PycharmProjects/APN/my_data/thumos14/annotations/thumos14_val_flow_list.txt', 'w') as f:
    f.writelines(lines)
