import pandas as pd
import numpy as np

def load_data():
    # 📂 Charger le bon fichier
    df = pd.read_csv("data/raw/historique.csv", sep=",")

    print("Colonnes:", df.columns)
    print("Shape original:", df.shape)

    # 🧹 Nettoyage colonnes inutiles
    if "name" in df.columns:
        df = df.drop(columns=["name"])

    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    # ⚠️ IMPORTANT : garder uniquement colonnes numériques
    df = df.select_dtypes(include=["number"])

    print("Shape après nettoyage:", df.shape)

    # 🧠 Prendre la dernière ligne
    data = df.iloc[-1].values

    print("Features récupérées:", len(data))

    # ⚠️ CORRECTION CRITIQUE : adapter à 15 features
    if len(data) < 15:
        # compléter avec des zéros
        data = np.pad(data, (0, 15 - len(data)), 'constant')
    elif len(data) > 15:
        # couper si trop de colonnes
        data = data[:15]

    print("Features finales:", len(data))

    # 🔁 reshape pour le modèle
    data = data.reshape(1, 1, 15)

    print("Shape final envoyé au modèle:", data.shape)

    return data
