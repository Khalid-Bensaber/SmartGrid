import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv("data/raw/historique.csv")

    df = df.select_dtypes(include=[np.number])

    data = df.iloc[-1].values

    data = data[:15]

    return data.astype(np.float32)
