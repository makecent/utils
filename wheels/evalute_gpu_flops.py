from urllib.request import urlopen

import torch
import timm
import torchvision.models as models
from PIL import Image
from torch.profiler import profile, record_function, ProfilerActivity
from fvcore.nn import FlopCountAnalysis, flop_count_table
from my_modules.backbone.i3d import I3D
from mmdet.registry import MODELS
import mmengine
import pytorchvideo


def count_and_print(model, inputs):
    flops = FlopCountAnalysis(model, inputs)
    print(flop_count_table(flops, max_depth=3))


def benchmark_run_time(model, inputs, gpu=False):
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if gpu else [ProfilerActivity.CPU]
    with profile(activities=activities, profile_memory=True,
                 record_shapes=True) as prof:
        with record_function(f"{model.__class__}_inference"):
            model(inputs)
    print(prof.key_averages().table(sort_by=f"{'cuda' if gpu else 'cpu'}_time_total", row_limit=10))


# Initialize the pseudo input
# pre_input = torch.rand(1, 1, 3, 192, 128, 128)  # By PlusTAD
# pre_input = torch.rand(1, 3, 32, 224, 224)    # by APN
pre_input = [torch.rand(1, 3, 96, 96, 96), torch.rand(1, 3, 384, 96, 96)]   # by E2E-TadTR

# Initialize the models
# cfg = mmengine.Config.fromfile("configs/repo_plustad_th14.py")
# model = MODELS.build(cfg.model)
# model = I3D()
model = torch.hub.load("facebookresearch/pytorchvideo", model='slowfast_r50', pretrained=True)

# Compare the inference time
count_and_print(model, pre_input)

# Compare the flop count
# benchmark_run_time(i3d, pre_input, gpu=True)