import flask
import hashlib
import os
from flask import Flask, url_for, request, render_template, redirect
from flask import make_response, jsonify
from flask import abort
from werkzeug.utils import secure_filename
from flask import session
from web_interface import controllers

app = Flask(__name__)
app.config.from_object(__name__)
app.config.update({
    "SECRET_KEY": b'|pl:=D\xde\x85\xdagjQ\xee\xfbc(\x8f\xfb\x10\xf9a*\xf1\x88',
    "TRAIN_DATA_FOLDER": r"d:\temp\train",
    "TEST_DATA_FOLDER": r"d:\temp\test",
    "VGG_19_NET_FILE_PATH": r"C:\Program Files\GraduationProject\VGG_19\imagenet-vgg-verydeep-19.mat",
    "STORE_TEMP_FEATURE_MAPS_PATH": r"C:\Users\anonymous\PycharmProjects\ufldl_tutorial\web_interface\static\temp"
})


def check_logged_in():
    if 'is_logged_in' not in session:
        abort(401)


@app.route('/')
@app.route('/index')
def index():
    if 'is_logged_in' not in session:
        return render_template('login.html')
    return render_template('index.html')


@app.route('/login', methods=['POST', 'GET'])
def login():
    return controllers.login_controller()


@app.route('/logout', methods=['POST'])
def logout():
    check_logged_in()
    return controllers.logout_controller()


@app.route('/save_files', methods=['POST'])
def save_files():
    check_logged_in()
    return controllers.save_file_controller()


@app.route("/analyze_data", methods=['POST'])
def analyze_data():
    check_logged_in()
    return controllers.analyze_data_controllers()


@app.route('/combine_data', methods=['POST'])
def combine_data():
    check_logged_in()
    return controllers.combine_data_controller()


@app.route('/show_feature_maps', methods=['POST'])
def show_feature_maps():
    check_logged_in()
    return controllers.show_feature_maps_controller()


@app.route("/classification", methods=["POST"])
def classification():
    check_logged_in()
    return controllers.classification_controller()


if __name__ == '__main__':
    app.run(host='localhost', debug=True)
