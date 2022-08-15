import copy
from os import path as osp

import numpy as np
from mmaction.datasets.builder import DATASETS
from mmaction.datasets.pipelines import Compose
from torch.utils.data import Dataset


@DATASETS.register_module()
class DenseExtracting(Dataset):
    """Dataset for densely feature extraction. Modified from APN dataset.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 clip_interval=4,
                 clip_len=128,
                 filename_tmpl='img_{:05}.jpg',
                 start_index=0,
                 modality='RGB'):
        super().__init__()
        ann_file = ann_file.split(',')
        video_name = ann_file[0]
        self.video_name = video_name if modality == 'Video' else video_name.rsplit('.', 1)[0]
        self.total_frames = int(ann_file[1])
        self.data_prefix = osp.realpath(data_prefix) if data_prefix is not None and osp.isdir(
            data_prefix) else data_prefix
        self.clip_interval = clip_interval
        self.clip_len = clip_len
        self.filename_tmpl = filename_tmpl
        self.start_index = start_index
        assert modality in ['RGB', 'Flow', 'Video']
        self.modality = modality
        self.pipeline = Compose(pipeline)

        self.frame_infos = self.load_video_info()

    def load_video_info(self):
        frame_infos = []
        video_path = osp.join(self.data_prefix, self.video_name)
        frame_inds = list(range(self.start_index, self.start_index + self.total_frames - self.clip_len + 1, self.clip_interval))
        self.feat_len = len(frame_inds)
        for frm_idx in frame_inds:
            frame_info = {'frame_index': frm_idx}
            if self.modality == 'Video':
                frame_info['filename'] = video_path
            else:
                frame_info['frame_dir'] = video_path
                frame_info['total_frames'] = self.total_frames
            frame_infos.append(frame_info)
        return frame_infos

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
