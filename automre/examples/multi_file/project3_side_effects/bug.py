import os


def run():
    mode = os.environ.get("APP_MODE")
    if mode == "strict":
        # Triggers ZeroDivisionError only when strict mode is enabled
        # (via config.py's module-level side effect).
        return 1 / 0
    return "ok"


def helper_never_called():
    return 42
