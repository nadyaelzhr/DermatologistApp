import streamlit as st
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)


import numpy as np
import json
import pickle
from PIL import Image
import tensorflow as tf
import joblib



# Load label map
with open("label_map.json", "r") as f:
    label_map = json.load(f)

# Load CNN model (.h5)
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("models/cnn_model.h5")

# Load Random Forest model (.pkl)
@st.cache_resource

def load_rf_model():
    return joblib.load("models/rf_model.pkl")


# Preprocessing image for CNN
def preprocess_image_cnn(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Preprocessing image for Random Forest
def preprocess_image_rf(image):
    image = image.resize((64, 64))
    img_array = np.array(image).flatten()  # 64x64x3 -> 12288
    return img_array.reshape(1, -1)

# Streamlit layout config
col1, col_mid, col2 = st.columns([6, 1, 6])

with col1:
    st.image("assets/logo.png", width=120)
    st.title("Dermatologist APP")
    st.write("Deteksi Penyakit Kulit menggunakan AI")

    uploaded_file = st.file_uploader("Upload Gambar Kulit", type=["jpg", "jpeg", "png"])
    model_choice = st.selectbox("Pilih Algoritma", ["CNN", "Random Forest"])
    predict_btn = st.button("Submit")

    with st.expander("Tentang Aplikasi Dermatologist"):
        st.write("""
            Aplikasi ini membantu pengguna mengidentifikasi potensi penyakit kulit melalui gambar.
            Pengguna dapat memilih metode prediksi menggunakan model CNN atau Random Forest,
            memberikan fleksibilitas sesuai kebutuhan. Aplikasi ini bersifat eksperimental dan
            tidak menggantikan diagnosis dokter secara langsung.
        """)

with col_mid:
    st.markdown(
        "<div style='border-left: 3px solid lightblue; height: 100vh;'></div>",
        unsafe_allow_html=True
    )

with col2:
    if predict_btn and uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.subheader("HASIL PREDIKSI")
        st.image(image, caption="Gambar yang Diupload", width=300)

        if model_choice == "CNN":
            model = load_cnn_model()
            input_image = preprocess_image_cnn(image)
            prediction = model.predict(input_image)
            class_idx = np.argmax(prediction)
            confidence = prediction[0][class_idx] * 100

        elif model_choice == "Random Forest":
            model = load_rf_model()
            input_image = preprocess_image_rf(image)
            class_idx = model.predict(input_image)[0]
            confidence = max(model.predict_proba(input_image)[0]) * 100

        class_name = label_map[str(class_idx)]
        st.write(f"**Nama Penyakit:** {class_name}")
        st.write(f"**Akurasi:** {confidence:.2f}%")
        st.write(f"**Algoritma yang digunakan:** {model_choice}")
        st.markdown("---")
        st.subheader("Deskripsi Penyakit")
        st.write(label_map.get(f"{class_idx}_desc", "Deskripsi belum tersedia."))
