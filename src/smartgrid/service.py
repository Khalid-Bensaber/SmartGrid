from src.smartgrid.model_loader import load_model
from src.smartgrid.predict import run_prediction

def predict_conso():
    model = load_model()
    result = run_prediction(model)

    return {
        "status": "ok",
        "prediction": result
    }
