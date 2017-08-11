import os
import pickle


def show_feature_maps_persistent(show_maps_path: str):
    _p = os.path.join(show_maps_path, "images.pydump")
    return pickle.load(open(_p, 'rb'))


def clean_temp_store_path(temp_store_path: str):
    """将*temp_store_path*下文件清空"""
    _temp_files = [os.path.join(temp_store_path, x) for x in os.listdir(temp_store_path)]
    for _ in _temp_files:
        os.remove(_)
