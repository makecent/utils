from mmcv.parallel import MMDataParallel
from mmaction.models import build_model
from mmaction.datasets import build_dataset, build_dataloader
from torch.nn.functional import cross_entropy
from fvcore.nn import parameter_count
from mvit import MViTB
import torch
import mmcv

train = True
model = torch.hub.load("facebookresearch/pytorchvideo", model='mvit_base_16x4', pretrained=False)
mmlab = False
param_num = True
# ===============================================================================================
model = MMDataParallel(model, device_ids=[0])


def yield_data(batch_size=4):
    for i in range(1000):
        yield dict(imgs=torch.randn(batch_size, 3, 16, 224, 224), label=torch.randint(400, size=(batch_size,)))


def single_gpu_test(model, mmlab=True):
    batch_size = 1
    model.eval()
    prog_bar = mmcv.ProgressBar(1000)
    for data in yield_data(batch_size):
        with torch.no_grad():
            if mmlab:
                result = model(return_loss=False, **data)
            else:
                result = model(data['imgs'].squeeze(1))

        # Assume result has the same length of batch_size
        # refer to https://github.com/open-mmlab/mmcv/issues/985
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()


def single_gpu_train(model, mmlab=True):
    batch_size = 4
    prog_bar = mmcv.ProgressBar(1000)
    for data in yield_data(batch_size):
        if mmlab:
            loss = model(**data)['loss_cls']
            loss.backward()
        else:
            result = model(data['imgs'].squeeze(1))
            loss = cross_entropy(result, data['label'].to(result.device))
            loss.backward()

        # Assume result has the same length of batch_size
        # refer to https://github.com/open-mmlab/mmcv/issues/985
        batch_size = len(data['imgs'])
        for _ in range(batch_size):
            prog_bar.update()


def main(train=True, mmlab=True, param_num=True):
    if param_num:
        print(parameter_count(model)[''])
    if train:
        single_gpu_train(model, mmlab=mmlab)
    else:
        single_gpu_test(model, mmlab=mmlab)


if __name__ == '__main__':
    main(train=train, mmlab=mmlab, param_num=param_num)
