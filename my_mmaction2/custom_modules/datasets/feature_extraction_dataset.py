import copy
from os import path as osp
from pathlib import Path

import numpy as np
from mmaction.registry import DATASETS
from mmengine.dataset import Compose
from torch.utils.data import Dataset
import cv2
import re


@DATASETS.register_module()
class DenseExtracting(Dataset):
    """Dataset for densely feature extraction. Modified from APN dataset.
    """

    def __init__(self,
                 video_name,
                 clip_len,
                 frame_interval,
                 clip_interval,
                 pipeline,
                 total_frames=None,
                 data_prefix=None,
                 filename_tmpl='img_{:05}.jpg',
                 start_index=0,
                 vformat='video',
                 modality='RGB'):
        super().__init__()
        self.video_name = video_name
        self.clip_len = clip_len
        self.frame_inteval = frame_interval
        self.clip_interval = clip_interval
        self.pipeline = Compose(pipeline)

        self.data_prefix = osp.realpath(data_prefix) if data_prefix is not None and osp.isdir(
            data_prefix) else data_prefix
        self.video_path = osp.join(self.data_prefix, self.video_name)

        if total_frames is None:
            if vformat == 'video':
                reader = cv2.VideoCapture(str(self.video_path))
                self.total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                pattern = make_regex_pattern(filename_tmpl)
                imgfiles = [img for img in Path(self.video_path).iterdir() if re.fullmatch(pattern, img.name)]
                if modality == 'RGB':
                    self.total_frames = len(imgfiles)
                else:
                    self.total_frames = len(imgfiles) // 2

        self.filename_tmpl = filename_tmpl
        self.start_index = int(start_index)
        assert vformat in ['video', 'rawframes']
        self.vformat = vformat
        self.modality = modality

        self.frame_infos = self.load_video_info()

    def load_video_info(self):
        frame_infos = []
        clip_starts = list(range(self.start_index,
                                 self.start_index + self.total_frames - self.clip_len * self.frame_inteval + 1,
                                 self.clip_interval))
        for c_start in clip_starts:
            frame_info = {'frame_inds': np.arange(c_start, c_start + self.clip_len) * self.frame_inteval,
                          'num_clips': 1,   # we input clips into the model one-by-one.
                          'clip_len': self.clip_len,
                          'label': 0}  # pseudo-label
            if self.vformat == 'video':
                frame_info['filename'] = self.video_path
            else:
                frame_info['frame_dir'] = self.video_path
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
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        return self.pipeline(results)

def make_regex_pattern(fixed_pattern):
    # Use regular expression to extract number of digits
    num_digits = re.search(r'\{:(\d+)\}', fixed_pattern).group(1)
    # Build the pattern string using the extracted number of digits
    pattern = fixed_pattern.replace('{:' + num_digits + '}', r'\d{' + num_digits + '}')
    # Replace {} with .
    pattern = pattern.replace('{}', '.')
    return pattern

