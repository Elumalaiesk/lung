from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from auth import login_required, send_auth_code_email
from db import get_prediction, init_db, insert_prediction, list_predictions
from predictor import run_ctgan_enhanced_prediction
import random


def create_app() -> Flask:
    app = Flask(__name__, instance_relative_config=True)

    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")
    app.config["AUTH_CODE"] = os.environ.get("AUTH_CODE", "123456")
    app.config["UPLOAD_FOLDER"] = os.environ.get(
        "UPLOAD_FOLDER", str(Path(app.root_path) / "uploads")
    )
    app.config["DATABASE"] = os.environ.get(
        "DATABASE", str(Path(app.instance_path) / "predictions.sqlite3")
    )

    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    init_db(app.config["DATABASE"])

    @app.context_processor
    def inject_globals():
        return {
            "user_email": session.get("user_email"),
            "now": datetime.utcnow(),
        }

    @app.get("/")
    def index():
        if session.get("user_email"):
            return redirect(url_for("dashboard"))
        return redirect(url_for("login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = (request.form.get("email") or "").strip().lower()
            code = (request.form.get("code") or "").strip()

            if not email:
                flash("Email is required.", "danger")
                return render_template("login.html")

            # If no code submitted, generate and send code
            if not code:
                generated_code = str(random.randint(100000, 999999))
                session["pending_email"] = email
                session["pending_code"] = generated_code
                try:
                    send_auth_code_email(email, generated_code)
                    flash("Auth code sent to your email. Please check your inbox.", "info")
                except Exception as e:
                    flash(f"Failed to send email: {e}", "danger")
                return render_template("login.html", email=email)

            # Validate submitted code
            if (
                email != session.get("pending_email") or
                code != session.get("pending_code")
            ):
                flash("Invalid auth code.", "danger")
                return render_template("login.html", email=email)

            session.pop("pending_email", None)
            session.pop("pending_code", None)
            session["user_email"] = email
            next_path = request.args.get("next")
            return redirect(next_path or url_for("dashboard"))

        return render_template("login.html")

    @app.post("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.get("/dashboard")
    @login_required
    def dashboard():
        recent = list_predictions(app.config["DATABASE"], limit=5)
        return render_template("dashboard.html", recent=recent)

    @app.route("/predict", methods=["GET", "POST"])
    @login_required
    def predict():
        if request.method == "POST":
            age_raw = (request.form.get("age") or "").strip()
            gender = (request.form.get("gender") or "").strip()
            age: Optional[int] = None
            if age_raw:
                try:
                    age = int(age_raw)
                    if age < 0 or age > 120:
                        raise ValueError()
                except ValueError:
                    flash("Age must be a number between 0 and 120.", "danger")
                    return render_template("predict.html")
            else:
                flash("Age is required for the current model.", "danger")
                return render_template("predict.html")

            if not gender:
                flash("Gender is required for the current model.", "danger")
                return render_template("predict.html")

            name = (request.form.get("name") or "").strip()
            if not name:
                flash("Patient name is required.", "danger")
                return render_template("predict.html")

            # Collect tabular clinical features required by the saved model.
            required_binary_fields = [
                "SMOKING",
                "YELLOW_FINGERS",
                "ANXIETY",
                "PEER_PRESSURE",
                "CHRONIC DISEASE",
                "FATIGUE",
                "ALLERGY",
                "WHEEZING",
                "ALCOHOL CONSUMING",
                "COUGHING",
                "SHORTNESS OF BREATH",
                "SWALLOWING DIFFICULTY",
                "CHEST PAIN",
            ]

            feature_payload: dict[str, object] = {
                "GENDER": gender,
                "AGE": age,
            }
            for f in required_binary_fields:
                raw = (request.form.get(f) or "").strip()
                try:
                    value = int(raw)
                except ValueError:
                    flash(f"{f} must be 1 (No) or 2 (Yes).", "danger")
                    return render_template("predict.html")
                if value not in (1, 2):
                    flash(f"{f} must be 1 (No) or 2 (Yes).", "danger")
                    return render_template("predict.html")
                feature_payload[f] = value

            filename = "TABULAR_INPUT"

            result = run_ctgan_enhanced_prediction(
                file_name=filename,
                age=age,
                gender=gender,
                features=feature_payload,
            )

            prediction_id = insert_prediction(
                db_path=app.config["DATABASE"],
                user_email=str(session.get("user_email")),
                file_name=filename,
                age=age,
                gender=gender or None,
                name=name,
                risk_probability=result.risk_probability,
                label=result.label,
                confidence=result.confidence,
                explanation=result.explanation,
            )

            return redirect(url_for("result", prediction_id=prediction_id))

        return render_template("predict.html")

    @app.get("/result/<int:prediction_id>")
    @login_required
    def result(prediction_id: int):
        row = get_prediction(app.config["DATABASE"], prediction_id)
        if row is None:
            abort(404)
        return render_template("result.html", p=row)

    @app.get("/history")
    @login_required
    def history():
        rows = list_predictions(app.config["DATABASE"], limit=None)
        return render_template("history.html", rows=rows)

    @app.get("/export")
    @login_required
    def export():
        rows = list_predictions(app.config["DATABASE"], limit=10)
        return render_template("export.html", rows=rows)

    @app.get("/export/csv")
    @login_required
    def export_csv():
        import csv
        import io

        rows = list_predictions(app.config["DATABASE"], limit=None)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["id", "created_at", "file_name", "age", "gender", "risk_probability", "label", "confidence"])
        for r in rows:
            writer.writerow(
                [
                    r.id,
                    r.created_at,
                    r.file_name,
                    r.age if r.age is not None else "",
                    r.gender or "",
                    f"{r.risk_probability:.4f}",
                    r.label,
                    f"{r.confidence:.4f}",
                ]
            )

        csv_bytes = buffer.getvalue().encode("utf-8")
        file_path = Path(app.instance_path) / "predictions_export.csv"
        file_path.write_bytes(csv_bytes)
        return send_file(
            file_path,
            as_attachment=True,
            download_name="predictions_summary.csv",
            mimetype="text/csv",
        )

    @app.get("/export/pdf/<int:prediction_id>")
    @login_required
    def export_pdf(prediction_id: int):
        row = get_prediction(app.config["DATABASE"], prediction_id)
        if row is None:
            abort(404)

        pdf_path = Path(app.instance_path) / f"prediction_{row.id}.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        width, height = letter

        left = 1.0 * inch
        y = height - 1.0 * inch

        c.setFont("Helvetica-Bold", 16)
        c.drawString(left, y, "Early-Stage Lung Cancer Prediction Report")

        y -= 0.4 * inch
        c.setFont("Helvetica", 10)
        c.drawString(left, y, f"Prediction ID: {row.id}")
        y -= 0.2 * inch
        c.drawString(left, y, f"Date (UTC): {row.created_at}")
        y -= 0.2 * inch
        c.drawString(left, y, f"User: {row.user_email}")

        y -= 0.35 * inch
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, "Input")
        y -= 0.25 * inch
        c.setFont("Helvetica", 10)
        c.drawString(left, y, "Input source: Tabular clinical features")
        y -= 0.2 * inch
        c.drawString(left, y, f"Patient Name: {row.name or '—'}")
        y -= 0.2 * inch
        c.drawString(left, y, f"Age: {row.age if row.age is not None else '—'}")
        y -= 0.2 * inch
        c.drawString(left, y, f"Gender: {row.gender or '—'}")

        y -= 0.35 * inch
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, "Output")
        y -= 0.25 * inch
        c.setFont("Helvetica", 10)
        c.drawString(left, y, f"Prediction (early-stage cancer): {row.label}")
        y -= 0.2 * inch
        c.drawString(left, y, f"Risk probability: {row.risk_probability * 100:.1f}%")
        y -= 0.2 * inch
        c.drawString(left, y, f"Confidence: {row.confidence * 100:.1f}%")

        y -= 0.35 * inch
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, "Explanation")
        y -= 0.25 * inch
        c.setFont("Helvetica", 10)

        text = c.beginText(left, y)
        text.setLeading(14)
        words = row.explanation.split()
        line = ""
        max_len = 95
        for w in words:
            candidate = (line + " " + w).strip()
            if len(candidate) > max_len:
                text.textLine(line)
                line = w
            else:
                line = candidate
        if line:
            text.textLine(line)

        c.drawText(text)

        c.showPage()
        c.save()

        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=f"prediction_{row.id}.pdf",
            mimetype="application/pdf",
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="127.0.0.1", port=3000)
