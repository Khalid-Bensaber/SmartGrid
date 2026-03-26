import numpy as np

def run_prediction(model):
    # 👉 données de test (tu peux changer plus tard)
    data = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

    # 👉 convertir en numpy
    data = np.array(data)

    # 👉 reshape pour correspondre au modèle (batch, time_steps, features)
    data = data.reshape(1, 1, -1)

    # 👉 prédiction
    prediction = model.predict(data)

    # 👉 convertir en liste pour JSON
    return prediction.tolist()
