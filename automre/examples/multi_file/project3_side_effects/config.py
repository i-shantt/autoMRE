import os

# Module-level side effect: sets an env var the bug logic reads.
os.environ["APP_MODE"] = "strict"

SETTINGS = {"threshold": 3}
