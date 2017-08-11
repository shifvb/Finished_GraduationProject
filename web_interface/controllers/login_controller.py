from flask import request, render_template, url_for, redirect, session
from flask import jsonify
from web_interface import services


def login_controller():
    """login controller"""
    # GET
    if request.method == 'GET':
        return render_template('login.html')
    # POST
    username = request.form['username']
    password = request.form['password']
    result_dict = dict()
    if services.login_service(username, password):
        session['is_logged_in'] = True
        session['username'] = username
        result_dict['result'] = True
    else:
        result_dict['result'] = False
        result_dict['reason'] = "Invalid username/password"
    return jsonify(result_dict)
