# Early-Stage Lung Cancer Prediction (CTGAN Project) — Simple Clinical UI

This is a lightweight **feature-based** web UI for early-stage lung cancer prediction.
It intentionally avoids image overlays/segmentation views and runs a **CTGAN + tabular ML** workflow.

## Pages / Flow

Login → Dashboard → Create Prediction → Result → History → Export

## Quick Start (Windows)

```powershell
cd e:\learning\Backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open: http://127.0.0.1:5000

## Login

- Email: anything
- Auth code: `123456` (change via environment variable `AUTH_CODE`)

## Notes

- Predictions are stored in SQLite at `instance/predictions.sqlite3`.
- The app will use your saved artifacts `lung_cancer_rf_model.pkl` + `label_encoders.pkl` (loaded via `joblib.load`) when they exist.
- Current saved model expects the 15 tabular inputs shown in the notebook (age, gender, and the 13 Yes/No clinical features). No DICOM upload is required for this model.
