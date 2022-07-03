import numpy as np
import pandas as pd
train_list = pd.read_csv("/home/louis/PycharmProjects/ProgPreTrain/my_data/kinetics400/kinetics400_train_list_videos.txt", header=None, delimiter=' ')
val_list = pd.read_csv("/home/louis/PycharmProjects/ProgPreTrain/my_data/kinetics400/kinetics400_val_list_videos.txt", header=None, delimiter=' ')

classes, count = np.unique(train_list[1], return_counts=True)
_, most_200_classes = zip(*sorted(zip(count, classes))[200:])

train_sampled_list = []
val_sampled_list = []
for new_class_id, class_id in enumerate(most_200_classes):
    train_cate_list = train_list[train_list[1] == class_id]
    train_cate_list[1] = new_class_id
    train_sampled_list.append(train_cate_list.sample(400))

    val_cate_list = val_list[val_list[1] == class_id]
    val_cate_list[1] = new_class_id
    val_sampled_list.append(val_cate_list.sample(25))

final_train_list = pd.concat(train_sampled_list).sample(frac=1)
final_val_list = pd.concat(val_sampled_list).sample(frac=1)

final_train_list.to_csv("/home/louis/PycharmProjects/ProgPreTrain/my_data/kinetics400/mini_kinetics200_train_list_videos.txt", header=False, index=False, sep=' ')
final_val_list.to_csv("/home/louis/PycharmProjects/ProgPreTrain/my_data/kinetics400/mini_kinetics200_val_list_videos.txt", header=False, index=False, sep=' ')