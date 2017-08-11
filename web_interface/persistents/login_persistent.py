import hashlib
from web_interface import persistents


def login_persistent(username: str, password: str) -> bool:
    db = persistents.get_connection()
    cursor = db.cursor()
    cursor.execute("""SELECT `password` FROM `users` WHERE `username`=%s;""", username)
    result = cursor.fetchone()
    if result:
        return _get_sha512_hexdigest(password) == result[0]
    else:
        return False


def _get_sha512_hexdigest(s: str) -> str:
    m = hashlib.sha512()
    m.update(s.encode(encoding='utf-8'))
    return m.hexdigest()
