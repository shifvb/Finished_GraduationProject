from web_interface import persistents
import random
import numpy as np
from PIL import Image
import os
import uuid


def show_feature_maps_service(show_maps_path: str, temp_store_path: str):
    # 清除temp_store_path文件夹内文件
    persistents.clean_temp_store_path(temp_store_path)

    # 从*show_maps_path*得到特征map文件
    arr = persistents.show_feature_maps_persistent(show_maps_path)

    # 随机生成若干图片，存于*show_maps_path*
    temp_paths = []
    for index in [random.randint(0, arr.shape[0] - 1) for _ in range(10)]:
        _img_data = _scale_img_data(arr[index])
        _img = Image.fromarray(_img_data, "L")
        _temp_name = uuid.uuid4().hex + ".jpg"
        temp_paths.append("/static/temp/" + _temp_name)
        _img.save(os.path.join(temp_store_path, _temp_name))
    del arr

    # 返回
    return temp_paths


def _scale_img_data(img_data: np.ndarray):
    """
    将 *img_data* scale 成[0, 255]的uint8
    :param img_data:
    :return:
    """
    img_data = img_data.reshape([64, 64])
    img_data = ((img_data / (img_data.max() - img_data.min()) + 1) / 2) * 255
    return img_data.astype(np.uint8)
