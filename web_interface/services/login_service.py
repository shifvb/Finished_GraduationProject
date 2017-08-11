from web_interface import persistents


def login_service(username: str, password: str) -> bool:
    """login_service"""
    return persistents.login_persistent(username, password)
