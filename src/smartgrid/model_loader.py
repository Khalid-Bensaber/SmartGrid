import tensorflow as tf
import os

def load_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(base_dir, "data", "model.h5")

    print("MODEL PATH:", model_path)

    model = tf.keras.models.load_model(model_path)
    return model
