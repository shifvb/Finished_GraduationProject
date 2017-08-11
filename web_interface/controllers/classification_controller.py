from web_interface import services
import os
from flask import jsonify, session
from web_interface.test_main import app


def classification_controller():
    try:
        train_folder = os.path.join(app.config['TRAIN_DATA_FOLDER'], session['username'])
        test_folder = os.path.join(app.config['TEST_DATA_FOLDER'], session['username'])
        accuracy = services.classification_service(train_folder, test_folder)
        return jsonify(result=True, data={'accuracy': ("%.3f%%" % (accuracy * 100))})
    except Exception as err:
        print(str(err))
        return jsonify(result=False)
