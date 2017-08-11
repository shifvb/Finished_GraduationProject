import os


def save_file_persistent(f, store_path: str, store_name: str):
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    f.save(os.path.join(store_path, store_name))
    return True
