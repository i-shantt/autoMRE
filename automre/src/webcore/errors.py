"""The one exception the job mechanics raise."""

from __future__ import annotations


class JobError(Exception):
    """A job could not be started or finished, with a reason fit to show.

    The distinction this carries is the one that matters to whoever is
    waiting: a JobError is something about *their* project or command — a
    test that does not exist, an archive that escapes its directory, a
    `pip install` that fails. Anything else reaching the caller is a bug
    in this tool and is reported as one, with its type name attached.

    It lives here rather than in a web layer because both front doors
    (the FastAPI worker and the in-sandbox runner) raise and catch it, and
    an exception class defined twice is two classes: `except JobError`
    in one process would not catch the other's.
    """
