import numpy as np
import pickle
import os


def combine_labels_persistent(labels: np.ndarray, out_path: str) -> None:
    pickle.dump(labels, open(out_path, 'wb'))


def is_already_done_labels_extract(extract_folder: str) -> bool:
    """
    检查是否已经做过特征提取了
    :param extract_folder: 检查文件夹路径
    :return:
    """
    return os.path.exists(os.path.join(extract_folder, "labels.pydump"))
