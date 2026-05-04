import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

# Load Model 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "cnn_cifar10.keras")

model = load_model(model_path)

# Label CIFAR-10
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# UI
st.set_page_config(page_title="CNN Image Classifier", layout="centered")

st.title("CNN Image Classifier")
st.write("Upload gambar untuk memprediksi objek menggunakan model CNN")

# Upload Image
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Input", width="stretch")

    # Preprocessing (AUTO DETECT)
    input_shape = model.input_shape[1:]

    if input_shape == (32, 32, 3):
        # CIFAR-10
        image_resized = image.resize((32, 32))
        img_array = np.array(image_resized) / 255.0

    elif input_shape == (28, 28, 1):
        # MNIST
        image = image.convert("L")
        image_resized = image.resize((28, 28))
        img_array = np.array(image_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)

    else:
        st.error(f"Model input shape tidak dikenali: {input_shape}")
        st.stop()

    # Tambah batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Debug info (optional)
    st.write("Model input shape:", model.input_shape)
    st.write("Image shape:", img_array.shape)

    # Prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    # Output
    if len(class_names) == prediction.shape[1]:
        predicted_class = class_names[predicted_index]
        st.success(f"Hasil Prediksi: **{predicted_class}**")
    else:
        st.success(f"Hasil Prediksi: Kelas {predicted_index}")

    st.write(f"Confidence: **{confidence:.2f}**")

    # Probabilities Chart
    st.subheader("Probabilitas Semua Kelas")

    prob_dict = {}
    for i in range(prediction.shape[1]):
        label = class_names[i] if i < len(class_names) else f"class_{i}"
        prob_dict[label] = float(prediction[0][i])

    st.bar_chart(prob_dict)

# Footer
st.markdown("---")
st.write("Dibuat dengan TensorFlow & Streamlit ")