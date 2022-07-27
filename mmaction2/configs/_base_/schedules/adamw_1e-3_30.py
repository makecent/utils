# optimizer
optimizer = dict(type='AdamW', lr=1e-3)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing',
                 min_lr=0,
                 warmup='linear',
                 warmup_ratio=0.1,
                 warmup_iters=2.5,
                 warmup_by_epoch=True)
total_epochs = 30
