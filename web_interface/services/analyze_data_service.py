from web_interface import persistents
import numpy as np


def analyze_data_service(analyze_path: str):
    all_labels_1, all_labels_2 = persistents.analyze_data_persistent(analyze_path)
    all_labels = np.concatenate([all_labels_1, all_labels_2], axis=0)
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
    all_labels_argmax = np.argmax(all_labels, axis=1)

    # 返回前台数据
    result_dict = dict()
    result_dict['img_num'] = str(all_labels.shape[0])
    result_dict['dtype'] = "uint8"
    result_dict['shape'] = "*".join([str(_) for _ in [224, 224, 3]])
    result_dict["liver_img_num"] = str((all_labels_argmax == 0).sum())
    result_dict["lymphoma_img_num"] = str((all_labels_argmax == 4).sum())
    result_dict["bg_img_num"] = str((all_labels_argmax == 5).sum())
    return result_dict
