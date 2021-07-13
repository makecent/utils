#%%  check if generated raw frames correct aligned
from pathlib import Path
from mmcv import imread
import numpy as np
p = Path("/home/louis/PycharmProjects/APN/data/ucf101/rawframes")
shape = []
for a in p.iterdir():
    for b in a.iterdir():
        i = list(b.glob("img_*"))[0]
        s = imread(i).shape
        shape.append(s)
        imgs = len(list(b.glob("img_*")))
        flow_x = len(list(b.glob("flow_x_*")))
        flow_y = len(list(b.glob("flow_y_*")))
        if flow_x == 0 or flow_y == 0 or imgs == 0:
            raise ValueError('c')
        if flow_x != flow_y:
            raise ValueError
        if imgs != flow_x+1:
            raise ValueError("b")
shape = np.array(shape)
u_shape = np.unique(shape, axis=0)



#%% Resize Images
from pathlib import Path
import mmcv
p = Path("data/DFMAD-70/Images/test")
for img_folder in p.iterdir():
    for img_p in img_folder.iterdir():
        new_path = list(img_p.parts)
        new_path[img_p.parts.index('test')] = 'resized_test'
        new_path = Path(*new_path)
        img = mmcv.imread(img_p)
        out_img = mmcv.imresize(img, (320, 180))
        mmcv.imwrite(out_img, f'{new_path}')

#%% change level 1 to level 2
from pathlib import Path
import pandas as pd
from mmcv import track_iter_progress
p = '/media/louis/louis-portable2/kinetics-dataset/train'
f = '/home/louis/PycharmProjects/APN/data/kinetics400/annotations/kinetics_train.csv'

p = Path(p)
f = pd.read_csv(f)

for i in track_iter_progress(list(p.iterdir())):
    id = '_'.join(i.stem.split('_')[:-2])
    label = f.loc[f['youtube_id'] == id, 'label'].to_list()
    if len(label) == 1:
        pass
    elif len(label) > 1:
        print(f"{id} got multiple labels: {label}")
    elif len(label) == 0:
        print(f"{id} got no labels")
    assert len(label) == 1
    label = label[0]
    p.joinpath(label).mkdir(exist_ok=True)
    i.rename(p.joinpath(label).joinpath(i.name))


#%% check resized videos
#find -name "*.mp4" -exec sh -c "echo '{}' >> errors.log; ffmpeg -v error -i '{}' -map 0:1 -f null - 2>> errors.log" \;
from pathlib import Path
import cv2
from mmcv import track_iter_progress
r = "/home/louis/PycharmProjects/APN/data/fineaction/webm_resized"
v = "/home/louis/PycharmProjects/APN/data/fineaction/videos_webm"
l = list(i.name for i in Path(v).iterdir())

def get_framenum(path):
    n = int(cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_COUNT))
    return n

for i in track_iter_progress(l):
    if 'mp4' in i:
        continue
    t1 = get_framenum(str(r + f'/{i.replace(".webm", ".mp4")}'))
    t2 = get_framenum(str(v + f'/{i}'))
    if t1 / t2 > 1.02 or t1 / t2 < 0.98:
        print('\n', i, t1, t2)
