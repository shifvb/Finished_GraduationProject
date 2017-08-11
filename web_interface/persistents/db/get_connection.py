import pymysql


def get_connection():
    return pymysql.connect(host='127.0.0.1', port=3306, database='test',
                           user='root', password='97033925')
