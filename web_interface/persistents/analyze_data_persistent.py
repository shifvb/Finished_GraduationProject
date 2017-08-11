import os
from scipy.io import loadmat


def analyze_data_persistent(analyze_path: str):
    """
    加载label
    :param analyze_path:
    :return:
    """
    all_labels_1 = loadmat(os.path.join(analyze_path, "allLabels1.mat"))['allLabels']
    all_labels_2 = loadmat(os.path.join(analyze_path, "allLabels2.mat"))["allLabels"]
    return all_labels_1, all_labels_2
