import numpy as np
from src.smartgrid.data_loader import load_data

def run_prediction(model):
    
    data = load_data()

    
    data = data.reshape(1, 1, -1)

   
    prediction = model.predict(data)

    return {
        "status": "ok",
        "prediction": prediction.tolist()
    }
