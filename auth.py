from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TypeVar

from flask import redirect, request, session, url_for
import smtplib
import os
from email.mime.text import MIMEText

from pathlib import Path

from dotenv import load_dotenv


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

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


def send_auth_code_email(recipient_email: str, code: str) -> None:
    """Send the authentication code to the user's email using SMTP settings from .env."""
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    smtp_use_tls = os.environ.get("SMTP_USE_TLS", "1") == "1"
    email_from = os.environ.get("EMAIL_FROM", smtp_user)
    if not smtp_user or not smtp_password:
        raise RuntimeError("SMTP_USER and SMTP_PASSWORD must be set in environment variables.")

    subject = "Your CTGAN Lung Prediction Auth Code"
    body = f"Your authentication code is: {code}\n\nPlease enter this code to log in."
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = recipient_email

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        if smtp_use_tls:
            server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(email_from, [recipient_email], msg.as_string())
