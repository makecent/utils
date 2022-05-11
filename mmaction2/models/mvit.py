import torch

from mmaction.models.builder import BACKBONES


@BACKBONES.register_module()
class MViTB(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        model = torch.hub.load("facebookresearch/pytorchvideo", model="mvit_base_16x4", pretrained=pretrained)
        delattr(model, "head")
        self.model = model

    def forward(self, x):
        return self.model(x)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        pass
