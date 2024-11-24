import tensorflow as tf
import os

input_model_dir = "models"
output_model_dir = "models/potatoes_model"

for version in [1, 2, 3]:
    keras_model = tf.keras.models.load_model(f"{input_model_dir}/{version}.keras")

    save_path = os.path.join(output_model_dir, str(version))
    keras_model.export(save_path)

    print(f"Exported models/{version}.keras to {save_path}")
