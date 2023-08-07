from mmaction.registry import MODELS
from mmengine.model import BaseModel
from torch.nn import AdaptiveAvgPool3d, Flatten

MODELS.register_module(module=AdaptiveAvgPool3d, name='AdaptiveAvgPool3d')
MODELS.register_module(module=Flatten, name='Flatten')


@MODELS.register_module()
class FeatExtractor(BaseModel):

    def __init__(self, backbone, neck=None, data_preprocessor=None):
        super().__init__()
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck) if neck is not None else None
        if data_preprocessor is None:
            data_preprocessor = dict(
                type='ActionDataPreprocessor',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                format_shape='NCTHW')
        self.data_preprocessor = MODELS.build(data_preprocessor)

    def forward(self,
                inputs,
                data_samples=None,
                mode='tensor'):
        num_crops = inputs.shape[1]
        inputs = inputs.view((-1,) + inputs.shape[2:])
        feat = self.backbone(inputs)
        if num_crops > 1:
            # testing augmentation
            feat = feat.view((-1, num_crops) + feat.shape[1:])
            feat = feat.mean(dim=1)
        if self.neck is not None:
            feat = self.neck(feat)
        return feat
