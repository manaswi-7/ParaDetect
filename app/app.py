# app/app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- Load model safely ---
MODEL_PATH = r"C:\Users\manas\OneDrive\Desktop\paradetect\model\malaria_model.h5"
model = load_model(MODEL_PATH)

# --- Streamlit UI setup ---
st.set_page_config(page_title="Malaria Cell Detection", page_icon="🧬", layout="centered")

st.title("🧬 Malaria Cell Detection App")
st.write("Upload a blood smear image, and the AI will tell you if it is Parasitized or Uninfected.")

# --- Sidebar instructions ---
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload a blood smear image.
2. Wait for the prediction.
3. Red = Parasitized
4. Green = Uninfected
""")

# --- File uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    # Open and display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # --- Preprocess ---
    img_resized = img.resize((64, 64))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # --- Prediction ---
    prediction = model.predict(img_array)
    
    # If binary sigmoid output (very common case)
    if prediction.shape[1] == 1:
        parasitized_prob = prediction[0][0]
        uninfected_prob = 1 - parasitized_prob
        pred_values = [parasitized_prob, uninfected_prob]
        class_index = 0 if parasitized_prob > 0.5 else 1
    else:
        pred_values = prediction[0]
        class_index = np.argmax(pred_values)
        
    class_names = ["Uninfected", "Parasitized"]
    confidence = pred_values[class_index] * 100

    # --- Show Prediction ---
    if class_index == 0:
        st.markdown(
            f"<h2 style='color:red'>Prediction: {class_names[class_index]} ({confidence:.2f}%)</h2>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<h2 style='color:green'>Prediction: {class_names[class_index]} ({confidence:.2f}%)</h2>",
            unsafe_allow_html=True
        )

    # --- Bar Chart (Corrected Properly) ---
    fig, ax = plt.subplots()

    bars = ax.bar(class_names, pred_values, color=['red', 'green'])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Confidence")

    # Add percentage labels
    for bar, value in zip(bars, pred_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.02,
            f"{value*100:.1f}%",
            ha='center'
        )

    st.pyplot(fig)
