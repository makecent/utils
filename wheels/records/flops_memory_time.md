# Backbone
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


# TAD models
Input shape: (1, 48, 224, 224)
PlusTAD: 8492 Mb
Mamba: 6173 Mb (6199 when accumulating 4 gradients)

Input shape: (1, 96, 224, 224)
PlusTAD: 8492 Mb
Mamba: 6173 Mb
