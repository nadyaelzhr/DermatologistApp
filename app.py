import streamlit as st
from PIL import Image
import numpy as np
import os
import time
from ultralytics import YOLO
from utils.preprocessing import resize_yolo, normalize_yolo

# ====== PATH & MODEL ======
MODEL_PATH = "models/yolo_model.pt"
LOGO_PATH = "assets/logo.png"

# ====== LABEL MAP (Internal) ======
label_map = {
  "Akiec": "AKIEC (Actinic Keratoses and Intraepithelial Carcinoma) adalah kondisi kulit yang merupakan gabungan antara actinic keratoses (AK) dan karsinoma in situ, yaitu kanker kulit tahap awal yang belum menyebar ke jaringan lebih dalam. Lesi AKIEC biasanya muncul sebagai plak tebal, kasar, dan bersisik dengan warna kemerahan atau kecokelatan, dan umumnya ditemukan di area kulit yang sering terpapar sinar matahari seperti wajah, kulit kepala, telinga, dan tangan. Gejala yang mungkin dirasakan meliputi gatal, nyeri ringan, atau bahkan luka yang mudah berdarah. Karena AKIEC sudah mengandung sel kanker tahap awal, pengobatannya lebih intensif dibanding AK biasa, seperti melalui eksisi bedah, kuretase dengan elektrokauter, terapi fotodinamik, atau penggunaan krim topikal khusus seperti 5-fluorouracil. Penanganan yang cepat dan tepat sangat penting untuk mencegah perkembangan menjadi karsinoma sel skuamosa invasif yang lebih berbahaya.",
  "Bcc": "Basal Cell Carcinoma (BCC) adalah jenis kanker kulit yang paling umum dan berasal dari sel basal, yaitu sel-sel di lapisan dasar epidermis. Kanker ini tumbuh perlahan dan jarang menyebar ke bagian tubuh lain, tetapi dapat merusak jaringan sekitarnya jika tidak segera ditangani. BCC biasanya disebabkan oleh paparan sinar ultraviolet (UV) yang berlebihan, baik dari sinar matahari maupun tanning bed. Ciri-cirinya meliputi benjolan kecil berwarna merah muda, keputihan, atau seperti mutiara yang mengkilap, dan sering kali tampak seperti luka yang tidak sembuh-sembuh, berkerak, atau berdarah. Lesi umumnya muncul di area yang sering terpapar sinar matahari seperti wajah, hidung, telinga, dan leher. Pengobatan BCC dapat dilakukan melalui eksisi bedah, kuretase dan elektrokauter, terapi fotodinamik, penggunaan krim topikal seperti imiquimod atau 5-fluorouracil, serta terapi radiasi untuk kasus tertentu. Pencegahan dapat dilakukan dengan menggunakan tabir surya, menghindari paparan sinar matahari langsung, dan rutin memeriksa kondisi kulit untuk deteksi dini.",
  "Df": "Dermatofibroma adalah tumor jinak pada kulit yang umumnya tidak berbahaya dan sering ditemukan pada orang dewasa, terutama wanita. Lesi ini muncul sebagai benjolan kecil yang keras, berwarna cokelat, merah muda, atau keabu-abuan, biasanya berdiameter kurang dari 1 cm dan sering terdapat di tungkai bawah, lengan, atau punggung. Ciri khas dermatofibroma adalah permukaannya yang menonjol atau agak cekung ke dalam jika ditekan (dimple sign), serta terasa kenyal atau padat saat diraba. Lesi ini biasanya tidak menimbulkan rasa sakit, tetapi bisa terasa gatal atau nyeri jika teriritasi. Pengobatan umumnya tidak diperlukan karena sifatnya jinak, namun bisa diangkat melalui pembedahan jika menyebabkan ketidaknyamanan, terus tumbuh, atau mengganggu secara kosmetik. Pemeriksaan kulit tetap penting untuk memastikan diagnosis dan membedakannya dari lesi kulit lain yang lebih serius.",
  "Nv": "Melanocytic Nevi, atau yang lebih dikenal sebagai tahi lalat, adalah pertumbuhan jinak pada kulit yang berasal dari sel melanosit, yaitu sel penghasil pigmen melanin. Lesi ini umumnya berwarna cokelat, hitam, atau kebiruan, dengan bentuk bulat atau oval, tepi yang teratur, dan permukaan yang halus atau sedikit menonjol. Melanocytic nevi dapat muncul sejak lahir (congenital nevi) atau berkembang seiring waktu (acquired nevi), dan sering ditemukan di area tubuh manapun. Meskipun sebagian besar tidak berbahaya, perubahan pada ukuran, warna, bentuk, atau gejala seperti gatal dan perdarahan perlu diwaspadai karena bisa menjadi tanda transformasi menuju melanoma. Pengobatan biasanya tidak diperlukan, tetapi tahi lalat dapat diangkat melalui pembedahan jika dicurigai ganas, mengganggu penampilan, atau sering teriritasi. Pemeriksaan rutin dan pemantauan perubahan pada tahi lalat penting untuk mendeteksi potensi risiko kanker kulit secara dini."
}
# bikin map lowercase agar pencocokan label tidak sensitif kapital
label_map_lower = {k.lower(): v for k, v in label_map.items()}

# ====== LOAD YOLO MODEL ======
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"Gagal memuat model YOLO: {e}")
    st.stop()

# ====== STYLING ======
st.markdown("""
    <style>
        .stImage > img {
            border: 2px solid #ddd;
            border-radius: 10px;
            background-color: white;
            padding: 6px;
        }
    </style>
""", unsafe_allow_html=True)

# ====== HEADER ======
col_logo, col_title = st.columns([1, 5])
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=150)
with col_title:
    st.markdown("<h1 style='margin-bottom:0;'>Dermatologist App - YOLOv8</h1>", unsafe_allow_html=True)
    st.markdown("<p style='margin-top:4px;color:#444;'>Deteksi dan klasifikasi penyakit kulit menggunakan YOLOv8.</p>", unsafe_allow_html=True)

st.write("---")
st.write("Unggah gambar kulit yang jelas (JPG/PNG). Klik **Submit** untuk melihat hasil deteksi dan klasifikasi.")

# ====== UPLOAD & SUBMIT ======
uploaded_file = st.file_uploader("📂 Unggah gambar kulit", type=["jpg", "jpeg", "png"])
submit = st.button("🔍 Submit untuk Prediksi YOLOv8")

if uploaded_file is not None and submit:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Gagal membuka gambar: {e}")
        st.stop()

    # ====== PREPROCESSING ======
    resized_img = resize_yolo(image, size=(640, 640))
    normalized_img = normalize_yolo(resized_img)

    # ====== PREDIKSI ======
    start_time = time.time()
    results = model.predict(np.array(normalized_img), imgsz=640, conf=0.25)
    end_time = time.time()

    if not results:
        st.warning("Tidak ada objek terdeteksi.")
        st.stop()

    # Ambil hasil deteksi pertama
    result = results[0]
    if len(result.boxes) > 0:
        class_id = int(result.boxes.cls[0])
        predicted_label = model.names[class_id]
    else:
        predicted_label = "Tidak terdefinisikan"

    # Simpan hasil deteksi dengan bounding box
    bbox_img_path = "deteksi_yolo.jpg"
    Image.fromarray(result.plot()).save(bbox_img_path)
    bbox_img = Image.open(bbox_img_path)

    # ====== TAMPILKAN GAMBAR ======
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.image(image, caption="Original", use_container_width=True)
    with c2:
        st.image(resized_img, caption="Resize 640x640", use_container_width=True)
    with c3:
        st.image(normalized_img, caption="Normalisasi", use_container_width=True)
    with c4:
        st.image(bbox_img, caption="Deteksi YOLOv8", use_container_width=True)

    # ====== HASIL PREDIKSI ======
    st.subheader("📌 Hasil Prediksi YOLOv8")
    st.markdown(f"**Kelas:** `{predicted_label}`")
    st.markdown(f"**Waktu Prediksi:** `{(end_time - start_time)*1000:.2f} ms`")

    # ====== DESKRIPSI (FULL & SCROLLABLE) ======
    desc = label_map_lower.get(str(predicted_label).lower(), "Deskripsi tidak ditemukan.")
    st.markdown(f"""
    <div style="
        background-color:#e8f4fd;
        padding:15px;
        border-radius:8px;
        border:1px solid #b3d7f2;
        max-height:200px;
        overflow-y:auto;
        font-size:16px;
        line-height:1.5;
        text-align:justify;
    ">
        {desc}
    </div>
    """, unsafe_allow_html=True)
