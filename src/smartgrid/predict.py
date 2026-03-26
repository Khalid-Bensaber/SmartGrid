import numpy as np

def run_prediction(model):
    data = np.random.rand(1, 15)
    prediction = model.predict(data)
    return prediction.tolist()
