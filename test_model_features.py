import joblib
import os

base_dir = "/root/Silver-Bullet-ML-BMAD"
model_path = os.path.join(base_dir, "models/s26_soft_fvg_ml_model.pkl")

if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("Model type:", type(model))
    if hasattr(model, "feature_names_in_"):
        print("Feature names in model:", list(model.feature_names_in_))
    else:
        print("Model does not have feature_names_in_")
else:
    print("Model not found")
