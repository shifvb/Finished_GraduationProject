from web_interface import persistents
import os
from scipy.io import loadmat
import numpy as np


def combine_labels_service(combine_folder: str) -> None:
    """
    从*combine_folder*中加载allLabels1.mat和allLabels2.mat，将其合并
    :param combine_folder: 用来加载labels的文件夹
    :return: None
    """
    persistents.combine_labels_persistent(*_gen_labels(combine_folder))


def is_already_done_labels_extract(extract_folder: str):
    return persistents.is_already_done_labels_extract(extract_folder)


def _gen_labels(parent_folder: str):
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
    # pickle.dump(all_labels, open(out_path, 'wb'))
    return all_labels, out_path
