import pandas as pd
import os

def load_data():
    # 📍 Chemin absolu propre
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    file_path = os.path.join(BASE_DIR, "data", "historique.csv")

    print("Chemin utilisé :", file_path)

    df = pd.read_csv(file_path, sep=",")

    print("Colonnes:", df.columns)
    print("Shape df:", df.shape)

    # supprimer colonnes inutiles
    if "name" in df.columns:
        df = df.drop(columns=["name"])
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    df = df.select_dtypes(include=["number"])

    print("Shape après nettoyage:", df.shape)

    data = df.iloc[-1].values
    data = data.reshape(1, 1, len(data))

    print("Shape final:", data.shape)

    return data
