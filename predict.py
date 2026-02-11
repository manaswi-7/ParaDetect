# predict.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import sys

# --- Load your trained model ---
MODEL_PATH = "model/malaria_model.h5"
model = load_model(MODEL_PATH)

# --- Load the test image ---
if len(sys.argv) < 2:
    print("Usage: python predict.py path_to_image")
    sys.exit()

img_path = sys.argv[1]
img = Image.open(img_path).convert("RGB")

# --- Preprocess the image ---
img = img.resize((64, 64))  # same size as training
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
img_array /= 255.0  # normalize if you trained with normalized data

# --- Prediction ---
prediction = model.predict(img_array)
class_index = np.argmax(prediction, axis=1)[0]
class_names = ["Parasitized", "Uninfected"]
confidence = prediction[0][class_index] * 100

print(f"Prediction: {class_names[class_index]} ({confidence:.2f}% confidence)")
