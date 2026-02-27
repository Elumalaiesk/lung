from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TypeVar

from flask import redirect, request, session, url_for

F = TypeVar("F", bound=Callable[..., Any])


def is_logged_in() -> bool:
    return bool(session.get("user_email"))


def login_required(fn: F) -> F:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any):
        if not is_logged_in():
            return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
