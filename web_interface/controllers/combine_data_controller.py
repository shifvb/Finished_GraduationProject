from web_interface import services
from flask import request, jsonify, session
from web_interface.test_main import app
import os


def combine_data_controller():
    is_combine_train_data = request.form['is_combine_train_data'] == 'true'
    # 决定使用train数据还是使用test数据进行合并
    if is_combine_train_data:
        combine_folder = app.config['TRAIN_DATA_FOLDER']
    else:
        combine_folder = app.config['TEST_DATA_FOLDER']
    combine_folder = os.path.join(combine_folder, session['username'])

    try:
        if services.is_already_done_labels_extract(combine_folder) and \
                services.is_already_done_images_extract(combine_folder):
            pass
        else:
            services.combine_labels_service(combine_folder)
            services.combine_images_service(combine_folder)
        return jsonify(result=True)
    except Exception as e:
        print(str(e))
        return jsonify(result=False)
