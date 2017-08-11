from web_interface.persistents.db.get_connection import get_connection

db = get_connection()

try:
    cursor = db.cursor()

    # 创建user表
    cursor.execute("""DROP TABLE IF EXISTS `users` """)
    cursor.execute("""CREATE TABLE `users`(
              `id` INT PRIMARY KEY NOT NULL,
              `username` VARCHAR(30),
              `password` VARCHAR(129)
            )""")
    cursor.execute("""INSERT INTO `users` (`id`, `username`, `password`) VALUES (%s, %s, %s)""", [
        0, 'shifvb',
        "55f07608aefbab2cbb5166458627125816c936e8af5c8ae02789c08d4ec54217010d4345fc1669be3074419076e14735e92fc7f9de53ac6ace4d97fac1aba0bb"
    ])
    cursor.execute("""INSERT INTO `users` (`id`, `username`, `password`) VALUES (%s, %s, %s)""", [
        1, 'admin',
        "55f07608aefbab2cbb5166458627125816c936e8af5c8ae02789c08d4ec54217010d4345fc1669be3074419076e14735e92fc7f9de53ac6ace4d97fac1aba0bb"
    ])
    db.commit()
except:
    db.rollback()
finally:
    db.close()
