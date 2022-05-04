from mmaction.datasets.pipelines import DecordInit
import os.path as osp
from tqdm import tqdm
video_reader = DecordInit()


def count_num_frames(filename):
    dummy = dict(filename=filename)
    return video_reader(dummy)['total_frames']


def get_ann_total_frames(mmaction_ann=None, data_prefix=None, output_ann=None):
    mmaction_ann = "/home/louis/PycharmProjects/ProgPreTrain/my_data/kinetics400/kinetics400_val_list_videos.txt"
    data_prefix = '/home/louis/PycharmProjects/ProgPreTrain/my_data/kinetics400/videos_val'
    output_ann = '/home/louis/PycharmProjects/ProgPreTrain/my_data/kinetics400/apn_kinetics400_val_video.txt'
    with open(mmaction_ann, 'r') as f:
        t = f.readlines()
    with open(output_ann, 'w') as f2:
        for line in tqdm(t):
            video_path = line.split()[0]
            absolute_path = osp.join(data_prefix, video_path)
            num_frames = count_num_frames(absolute_path)
            f2.write(f"{video_path} {num_frames}\n")
get_ann_total_frames()
