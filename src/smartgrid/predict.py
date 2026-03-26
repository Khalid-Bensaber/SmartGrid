import numpy as np
from src.smartgrid.data_loader import load_data

def run_prediction(model):
    # 👉 récupérer les données
    data = load_data()

    # 👉 reshape pour modèle
    data = data.reshape(1, 1, -1)

    # 👉 prédiction
    prediction = model.predict(data)

    return {
        "status": "ok",
        "prediction": prediction.tolist()
    }
