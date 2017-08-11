import os
import pickle

import h5py
import numpy as np
import tensorflow as tf
from scipy.io import loadmat

from lymph.vgg19 import net


def gen_labels(parent_folder: str):
    # 生成文件路径
    path_1 = os.path.join(parent_folder, "allLabels1.mat")
    path_2 = os.path.join(parent_folder, "allLabels2.mat")
    out_path = os.path.join(parent_folder, "labels.pydump")
    # 从文件中加载matlab文件并拼接
    all_labels_1 = loadmat(path_1)['allLabels']
    all_labels_2 = loadmat(path_2)["allLabels"]
    all_labels = np.concatenate([all_labels_1, all_labels_2], axis=0)
    print("[DEBUG] all_labels_1: dtype={}, shape={}".format(all_labels_1.dtype, all_labels_1.shape))
    print("[DEBUG] all_labels_2: dtype={}, shape={}".format(all_labels_2.dtype, all_labels_2.shape))
    print("[DEBUG] all_labels:   dtype={}, shape={}".format(all_labels.dtype, all_labels.shape))
    del all_labels_1, all_labels_2
    # 对标签进行处理
    for i in range(all_labels.shape[0]):
        temp_list = [0, 0, 0, 0, 0, 0]
        if (all_labels[i][:5] == 0).all():  # 如果前面都是0，那么就是背景
            temp_list[5] = 1
        else:
            temp_list[np.argmax(all_labels[i][:5])] = 1  # 如果前面不都是0，那么就是相应的种类(只有一种)
        all_labels[i] = temp_list
    # 分析各种类别数量
    _class_map = {k: "({:})".format(v) for k, v in {0: "肝", 1: "左肾", 2: "右肾", 3: "膀胱", 4: "淋巴瘤", 5: "背景"}.items()}
    total_num = all_labels.shape[0]
    all_labels_argmax = np.argmax(all_labels, axis=1)
    for _class in range(all_labels.shape[1]):
        _class_sum = (all_labels_argmax == _class).sum()
        print("[DEBUG] class{:1} {:4}\t\t: {:3}/{} - {:.3}%".format(_class, _class_map[_class], _class_sum,
                                                                    total_num, (_class_sum / total_num) * 100))
    # 转换为uint8, 存盘
    all_labels = all_labels.astype(np.uint8)
    print("[DEBUG] all_labels:   dtype={}, shape={}".format(all_labels.dtype, all_labels.shape))
    pickle.dump(all_labels, open(out_path, 'wb'))


def gen_images(parent_folder: str):
    # 生成文件路径
    path_1 = os.path.join(parent_folder, "allParts1.mat")
    path_2 = os.path.join(parent_folder, "allParts2.mat")
    out_path = os.path.join(parent_folder, "images.pydump")
    # 从文件中加载matlab文件并拼接
    with h5py.File(path_1) as f:
        all_parts_1 = np.array(f['allParts'], dtype=np.uint8)
    with h5py.File(path_2) as f:
        all_parts_2 = np.array(f['allParts'], dtype=np.uint8)
    all_parts = np.concatenate([all_parts_1, all_parts_2], axis=0)
    all_parts = all_parts.transpose([0, 2, 1])
    print("[DEBUG] all_parts_1: shape={}, dtype={}".format(all_parts_1.shape, all_parts_1.dtype))
    print("[DEBUG] all_parts_2: shape={}, dtype={}".format(all_parts_2.shape, all_parts_2.dtype))
    print("[DEBUG] all_parts:   shape={}, dtype={}".format(all_parts.shape, all_parts.dtype))
    del all_parts_1, all_parts_2
    # 下一步骤就是放到vgg19-f里，生成权值
    images = all_parts
    print("[DEBUG] images:      shape={}, dtype={}".format(images.shape, images.dtype))
    # 生成特征值
    feature_list = []
    batch_size = 20
    with tf.Session() as sess:
        for i in range(0, images.shape[0], batch_size):
            batch_xs = images[i: i + batch_size]
            batch_xs = np.stack([batch_xs, batch_xs, batch_xs], axis=3) - mean_pixel
            if i % 1000 == 0:
                print("[DEBUG] processing {}-{}...".format(i, i + batch_xs.shape[0]))
            result = sess.run(nets['fc1'], feed_dict={input_images: batch_xs})
            result = result.reshape(batch_xs.shape[0], 4096)
            feature_list.append(result)
    # 存盘
    arr = np.concatenate(feature_list, axis=0)
    print("[DEBUG] arr: shape={}, dtype={}".format(arr.shape, arr.dtype))
    pickle.dump(arr, open(out_path, 'wb'))


if __name__ == '__main__':
    # 加载vgg19-f网络
    net_path = r"C:\VGG_19\imagenet-vgg-verydeep-19.mat"
    input_images = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name="input")
    nets, mean_pixel, _ = net(net_path, input_images)
    # 主程序开始
    print("-" * 30, "PT06535", '-' * 30)
    gen_labels(parent_folder=r"C:\LYMPH_data\mat224dataUintPT06535")
    gen_images(parent_folder=r"C:\LYMPH_data\mat224dataUintPT06535")
    print("-" * 30, "PT06548", '-' * 30)
    gen_labels(parent_folder=r"C:\LYMPH_data\mat224dataUintPT06548")
    gen_images(parent_folder=r"C:\LYMPH_data\mat224dataUintPT06548")
    print("-" * 30, "PT06586", '-' * 30)
    gen_labels(parent_folder=r"C:\LYMPH_data\mat224dataUintPT06586")
    gen_images(parent_folder=r"C:\LYMPH_data\mat224dataUintPT06586")
    print("-" * 30, "PT38875", '-' * 30)
    gen_labels(parent_folder=r"C:\LYMPH_data\mat224dataUintPT38875")
    gen_images(parent_folder=r"C:\LYMPH_data\mat224dataUintPT38875")
    print("-" * 30, "PT38904", '-' * 30)
    gen_labels(parent_folder=r"C:\LYMPH_data\mat224dataUintPT38904")
    gen_images(parent_folder=r"C:\LYMPH_data\mat224dataUintPT38904")
