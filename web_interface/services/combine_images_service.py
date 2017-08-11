import os
import numpy as np
import h5py
import tensorflow as tf

from web_interface import persistents


def combine_images_service(combine_folder: str) -> None:
    """
    从*combine_folder*中加载allParts1.mat和allParts2.mat，将其合并，同时生成vgg-19net特征值
    :param combine_folder: 用来加载images的文件夹
    :return: None
    """
    nets, mean_pixel, input_images = persistents.load_vgg19net()
    images, out_path = _gen_images(nets, mean_pixel, input_images, combine_folder)
    persistents.release_vgg19net()
    persistents.combine_images_persistent(images, out_path)


def is_already_done_images_extract(extract_folder: str):
    return persistents.is_already_done_images_extract(extract_folder)


def _gen_images(nets, mean_pixel, input_images, parent_folder: str):
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
    # pickle.dump(arr, open(out_path, 'wb'))
    return arr, out_path
