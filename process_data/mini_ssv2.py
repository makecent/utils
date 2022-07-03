import numpy as np
import pandas as pd
train_list = pd.read_csv("/home/louis/PycharmProjects/APN/my_data/sthv2/sthv2_train_list_rawframes.txt", header=None, delimiter=' ')
val_list = pd.read_csv("/home/louis/PycharmProjects/APN/my_data/sthv2/sthv2_val_list_rawframes.txt", header=None, delimiter=' ')

classes, count = np.unique(train_list[2], return_counts=True)
_, most_87_classes = zip(*sorted(zip(count, classes))[87:])

train_sampled_list = []
val_sampled_list = []
for category in most_87_classes:
    train_cate_list = train_list[train_list[2] == category]
    train_sampled_list.append(train_cate_list.sample(600))

    val_cate_list = val_list[val_list[2] == category]
    val_sampled_list.append(val_cate_list.sample(35))

final_train_list = pd.concat(train_sampled_list).sample(frac=1)
final_val_list = pd.concat(val_sampled_list).sample(frac=1)

final_train_list.to_csv("/home/louis/PycharmProjects/APN/my_data/sthv2/mini_sthv2_train_list_rawframes.txt", header=False, index=False)
final_val_list.to_csv("/home/louis/PycharmProjects/APN/my_data/sthv2/mini_sthv2_val_list_rawframes.txt", header=False, index=False)