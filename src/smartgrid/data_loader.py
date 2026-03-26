import pandas as pd

def load_data():
    # 🔹 Charger le bon fichier
    df = pd.read_csv("data/historique.csv", sep=",")

    # 🔍 DEBUG (tu peux enlever après)
    print("Colonnes:", df.columns)
    print("Shape df:", df.shape)

    # 🔹 Supprimer colonnes inutiles
    if "name" in df.columns:
        df = df.drop(columns=["name"])
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    # 🔹 Garder seulement les colonnes numériques
    df = df.select_dtypes(include=["number"])

    print("Shape après nettoyage:", df.shape)

    # 🔹 Prendre dernière ligne
    data = df.iloc[-1].values

    # 🔹 Adapter au modèle (15 features attendues)
    data = data.reshape(1, 1, len(data))

    print("Shape final envoyé au modèle:", data.shape)

    return data
