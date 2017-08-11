from web_interface import services
from flask import request, jsonify, session
from web_interface.test_main import app
import os


def show_feature_maps_controller():
    # 获得show_map_math
    if request.form['is_show_train_data_feature_maps'] == 'true':
        show_maps_path = app.config['TRAIN_DATA_FOLDER']
    else:
        show_maps_path = app.config['TEST_DATA_FOLDER']
    show_maps_path = os.path.join(show_maps_path, session['username'])
    temp_store_path = app.config['STORE_TEMP_FEATURE_MAPS_PATH']

    # 判断是不是已经取过feature了
    if not (services.is_already_done_images_extract(show_maps_path) and
                services.is_already_done_labels_extract(show_maps_path)):
        return jsonify(result=False)

    # 随机提取一些特征图，得到
    img_urls = services.show_feature_maps_service(show_maps_path, temp_store_path)

    return jsonify(result=True, data={
        "img_urls": img_urls,
    })
