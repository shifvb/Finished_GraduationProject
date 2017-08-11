import numpy as np
import pickle
import os


def combine_images_persistent(images: np.ndarray, out_path: str) -> None:
    pickle.dump(images, open(out_path, 'wb'))


def is_already_done_images_extract(extract_folder: str) -> bool:
    """
    检查是否已经做过特征提取了
    :param extract_folder: 检查文件夹路径
    :return:
    """
    return os.path.exists(os.path.join(extract_folder, "images.pydump"))
