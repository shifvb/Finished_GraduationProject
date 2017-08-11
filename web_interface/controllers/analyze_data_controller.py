from web_interface import services
from flask import request, jsonify, session
from web_interface.test_main import app
import os


def analyze_data_controllers():
    # generate analyze folder
    if request.form['is_analyze_train_data'] == 'true':
        analyze_path = app.config['TRAIN_DATA_FOLDER']
    else:
        analyze_path = app.config['TEST_DATA_FOLDER']
    analyze_path = os.path.join(analyze_path, session['username'])

    try:
        result_data = services.analyze_data_service(analyze_path)
        return jsonify(result=True, data=result_data)
    except Exception as e:
        print(str(e))
        return jsonify(result=False)
