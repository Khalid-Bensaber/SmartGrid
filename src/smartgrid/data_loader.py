import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv("data/raw/historique.csv", sep=",")

    print("Colonnes:", df.columns)
    print("Shape original:", df.shape)
    
    if "name" in df.columns:
        df = df.drop(columns=["name"])

    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    df = df.select_dtypes(include=["number"])

    print("Shape après nettoyage:", df.shape)

    data = df.iloc[-1].values

    print("Features récupérées:", len(data))

    if len(data) < 15:
        
        data = np.pad(data, (0, 15 - len(data)), 'constant')
    elif len(data) > 15:
        
        data = data[:15]

    print("Features finales:", len(data))

    
    data = data.reshape(1, 1, 15)

    print("Shape final envoyé au modèle:", data.shape)

    return data
