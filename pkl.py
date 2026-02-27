import joblib


model = joblib.load("lung_cancer_rf_model.pkl")
encoders = joblib.load("label_encoders.pkl")

print("model:", type(model))
print("n_features_in_:", getattr(model, "n_features_in_", None))
print("feature_names_in_:", getattr(model, "feature_names_in_", None))
print("encoders:", type(encoders), "keys=", list(encoders.keys()) if hasattr(encoders, "keys") else None)