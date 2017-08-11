from flask import session,jsonify


def logout_controller():
    try:
        session.pop("is_logged_in", None)
        session.pop("username", None)
        return jsonify(result=True)
    except Exception as e:
        print(str(e))
        return jsonify(result=False)
