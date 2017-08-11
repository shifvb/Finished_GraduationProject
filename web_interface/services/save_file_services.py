from web_interface import persistents


def save_file_services(f, store_path: str, store_name: str):
    return persistents.save_file_persistent(f, store_path, store_name)
