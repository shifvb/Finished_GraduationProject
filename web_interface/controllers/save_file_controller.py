from web_interface import services
from web_interface.test_main import app
from flask import request, jsonify, session
import os


def save_file_controller():
    """将文件临时存储到本地磁盘上"""
    try:
        f = request.files['file']
        is_train_data = request.form['is_train_data']
        is_image_data = request.form['is_image_data']
        part_num = request.form['part_num']
        # generate store path
        if is_train_data == 'true':
            store_path = app.config['TRAIN_DATA_FOLDER']
        else:
            store_path = app.config['TEST_DATA_FOLDER']
        store_path = os.path.join(store_path, session['username'])
        # generate store name
        store_name = "allParts{}.mat" if is_image_data == 'true' else "allLabels{}.mat"
        store_name = store_name.format(part_num)
        services.save_file_services(f, store_path, store_name)
        return jsonify(result=True)
    except Exception as e:
        print(str(e))
        return jsonify(result=False)
