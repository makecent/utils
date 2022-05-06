Environment:
```
scratch:
    mmaction2
    my_models
        __init__.py
        resnet3d_sony.py
    i3d.py
    x3d.py
    i3d_sony.py
        # imports
        custom_imports = dict(imports=['my_models'], allow_failed_imports=False)
        
        model = dict(
            type='Recognizer3D',
            backbone=dict(
                type='ResNet3d_sony',
                modality='rgb'),
            cls_head=dict(
                type='I3DHead',
                num_classes=400,
                in_channels=1024,
                spatial_type='avg',
                dropout_ratio=0.5,
                init_std=0.01),
            # model training and testing settings
            train_cfg=None,
            test_cfg=dict(average_clips='prob'))
```

Command used:
```shell
PYTHONPATH=$PWD:$PYTHONPATH python mmaction2/tools/analysis/get_flops.py x3d.py --shape 1 3 16 224 224
PYTHONPATH=$PWD:$PYTHONPATH python mmaction2/tools/analysis/get_flops.py i3d.py --shape 1 3 16 224 224
PYTHONPATH=$PWD:$PYTHONPATH python mmaction2/tools/analysis/get_flops.py i3d_sony.py --shape 1 3 16 224 224
```


x3d:
==============================
Input shape: (1, 3, 16, 224, 224)
Flops: 4.97 GFLOPs
Params: 3.79 M
==============================

i3d:
==============================
Input shape: (1, 3, 16, 224, 224)
Flops: 16.68 GFLOPs
Params: 28.04 M
==============================

i3d_sony:
==============================
Input shape: (1, 3, 16, 224, 224)
Flops: 27.86 GFLOPs
Params: 12.7 M
==============================

Interesting, it seems that the x3d is the smallest model, at least smaller than i3d_sony.
But when I use x3d backbone in APN, the batch size have to be reduced to **8** to avoid OOM error, 
which can be up to **20** when using i3d_sony backbone under the same setting (two 1080 ti). 


