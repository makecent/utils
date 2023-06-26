from fvcore.nn import parameter_count_table
from mmdet.registry import MODELS, DATASETS
from mmengine import Config
from mmengine.runner.checkpoint import load_state_dict, _load_checkpoint

cfg = Config.fromfile("configs/tadtr.py")

model = cfg.model
model._scope_ = 'mmdet'
model.bbox_head.active = False
model = MODELS.build(model)
model.init_weights()
print(parameter_count_table(model, max_depth=3))

# Load state_dict
state_dict = _load_checkpoint("model_init.pth", map_location='cpu')
# state_dict = state_dict['model']

# Remove the transformer. prefix
state_dict = {
    k[len('transformer.'):] if k.startswith('transformer.') else k: v
    for k, v in state_dict.items()
}
# Remove the segment_embed which is repeated (on both transformer and transformer.decoder)
old_state_dict_keys = list(state_dict.keys())
for key in old_state_dict_keys:
    if 'decoder.segment_embed' in key:  #
        state_dict.pop(key)
state_dict['neck.convs.0.conv.weight'] = state_dict.pop('input_proj.0.0.weight')[..., None]
state_dict['neck.convs.0.conv.bias'] = state_dict.pop('input_proj.0.0.bias')

# Special handling before the main loop
temp_dict = {}
# special_keys = ['sampling_offsets', 'reference_points']
for old_key in list(state_dict.keys()):  # Using list to avoid runtime error due to change in dict size
    # if any(special_key in old_key for special_key in special_keys):
    #     expanded = state_dict[old_key].repeat_interleave(2, dim=0)
    #     expanded[1::2] = 0
    #     state_dict[old_key] = expanded

    # Special handling for "self_attn" to "self_attn.attn" when "self_attn" and "decoder" are both substrings
    if 'self_attn' in old_key and 'decoder' in old_key:
        new_key = old_key.replace('self_attn', 'self_attn.attn')
        state_dict[new_key] = state_dict.pop(old_key)

    # Exchange 'norm1' and 'norm2' in 'decoder'.
    if 'decoder' in old_key and 'norm' in old_key:
        if 'norm1' in old_key:
            new_key = old_key.replace('norm1', 'norm2')
            temp_dict[new_key] = state_dict.pop(old_key)
        elif 'norm2' in old_key:
            new_key = old_key.replace('norm2', 'norm1')
            temp_dict[new_key] = state_dict.pop(old_key)

state_dict.update(temp_dict)
key_mapping = {
    'input_proj.0.1': 'neck.convs.0.gn',
    'segment_embed.0.layers.0': 'bbox_head.bbox_head.reg_branches.0.0',
    'segment_embed.0.layers.1': 'bbox_head.bbox_head.reg_branches.0.2',
    'segment_embed.0.layers.2': 'bbox_head.bbox_head.reg_branches.0.4',
    'segment_embed.1.layers.0': 'bbox_head.bbox_head.reg_branches.1.0',
    'segment_embed.1.layers.1': 'bbox_head.bbox_head.reg_branches.1.2',
    'segment_embed.1.layers.2': 'bbox_head.bbox_head.reg_branches.1.4',
    'segment_embed.2.layers.0': 'bbox_head.bbox_head.reg_branches.2.0',
    'segment_embed.2.layers.1': 'bbox_head.bbox_head.reg_branches.2.2',
    'segment_embed.2.layers.2': 'bbox_head.bbox_head.reg_branches.2.4',
    'segment_embed.3.layers.0': 'bbox_head.bbox_head.reg_branches.3.0',
    'segment_embed.3.layers.1': 'bbox_head.bbox_head.reg_branches.3.2',
    'segment_embed.3.layers.2': 'bbox_head.bbox_head.reg_branches.3.4',
    'class_embed': 'bbox_head.bbox_head.cls_branches',
    'linear1': 'ffn.layers.0.0',
    'linear2': 'ffn.layers.1',
    'norm1': 'norms.0',
    'norm2': 'norms.1',
    'norm3': 'norms.2',
    'reference_points': 'reference_points_fc',
    'query_embed': 'query_embedding'
}

old_state_dict_keys = list(state_dict.keys())
for old_key in old_state_dict_keys:
    for old_substr, new_substr in key_mapping.items():
        if old_substr in old_key:
            new_key = old_key.replace(old_substr, new_substr)
            state_dict[new_key] = state_dict.pop(old_key)
            break

# import torch
# torch.save(state_dict, "tadtr_init.pth")

load_state_dict(model, state_dict, strict=False)

train_ds = cfg.train_dataloader.dataset
val_ds = cfg.val_dataloader.dataset
train_ds._scope_ = 'mmdet'
val_ds._scope_ = 'mmdet'

train_ds = DATASETS.build(train_ds)
data = train_ds[0]
data['inputs'] = [data['inputs']]
data['data_samples'] = [data['data_samples']]

model.eval()
data = model.data_preprocessor(data, True)
results = model.loss(data['inputs'], data['data_samples'])

print('end')

# from mmengine.runner.checkpoint import load_state_dict, _load_checkpoint
# state_dict = _load_checkpoint('outputs/thumos14_i3d2s_tadtr/model_best.pth', map_location='cpu')
# load_state_dict(model, state_dict['model'])