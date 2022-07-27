import copy

from os import path as osp
from mmaction.datasets.builder import DATASETS
from mmaction.datasets.pipelines import Compose
from torch.utils.data import Dataset
import numpy as np


@DATASETS.register_module()
class DenseExtracting(Dataset):
    """Dataset for densely feature extraction. Modified from APN dataset.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 clip_interval=4,
                 filename_tmpl='img_{:05}.jpg',
                 start_index=0,
                 modality='RGB'):
        super().__init__()
        self.ann_file = ann_file
        self.data_prefix = osp.realpath(data_prefix) if data_prefix is not None and osp.isdir(
            data_prefix) else data_prefix
        self.clip_interval = clip_interval
        self.filename_tmpl = filename_tmpl
        self.start_index = start_index
        assert modality in ['RGB', 'Flow', 'Video']
        self.modality = modality
        self.pipeline = Compose(pipeline)
        self.video_infos = self.read_ann_files()
        self.video_len = [0]
        self.frame_infos = self.load_video_info()

    def read_ann_files(self):
        video_infos = {}
        with open(self.ann_file, 'r') as fin:
            for line in fin.readlines():
                line_split = line.strip().split(',')
                video_name = str(line_split[0])
                total_frames = int(line_split[1])
                if self.modality != 'Video':
                    # remove the suffix, e.g. '.mp4', if not using video format, e.g. RGB frames
                    video_name = video_name.rsplit('.', 1)[0]
                video_infos.setdefault(video_name, total_frames)

        return video_infos

    def load_video_info(self):
        frame_infos = []
        for video_name, total_frames in self.video_infos.items():
                video_name = osp.join(self.data_prefix, str(video_name))
                frame_inds = list(range(self.start_index, self.start_index + total_frames, self.clip_interval))
                self.video_len.append(len(frame_inds))
                for frm_idx in frame_inds:
                    frame_info = {'frame_index': frm_idx}
                    if self.modality == 'Video':
                        frame_info['filename'] = video_name
                    else:
                        frame_info['frame_dir'] = video_name
                        frame_info['total_frames'] = total_frames
                    frame_infos.append(frame_info)
        return frame_infos

    def split_results_by_video(self, results):
        results_vs_video = []
        it = np.cumsum(self.video_len)
        for i, j in zip(it[:-1], it[1:]):
            results_by_video = results[i: j]
            results_vs_video.append(results_by_video)
        return results_vs_video

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.frame_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        results = copy.deepcopy(self.frame_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = 'RGB' if self.modality == 'Video' else self.modality
        results['start_index'] = self.start_index
        return self.pipeline(results)