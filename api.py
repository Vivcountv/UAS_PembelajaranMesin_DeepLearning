import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import keras

# ==========================================
# 1. DEFINISI CUSTOM OBJECTS (WAJIB ADA)
# ==========================================
# Agar Streamlit tidak error saat meload model Siamese
@keras.saving.register_keras_serializable()
def euclidean_distance(vectors):
    x, y = vectors
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

@keras.saving.register_keras_serializable()
class EuclideanDistance(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        x, y = inputs
        return tf.math.reduce_sum(tf.math.square(x - y), axis=-1, keepdims=True)

# ==========================================
# 2. CONFIG & LOAD MODEL
# ==========================================
st.set_page_config(page_title="Sistem Keamanan Biometrik", page_icon="🖐️", layout="wide")

@st.cache_resource
def load_models():
    # A. Load Model Klasifikasi
    try:
        model_clf = keras.models.load_model("model_klasifikasi_telapak.h5", compile=False)
    except Exception as e:
        st.error(f"Gagal memuat Model Klasifikasi: {e}")
        model_clf = None

    # B. Load Model Siamese (Verifikasi)
    custom_objects = {
        'EuclideanDistance': EuclideanDistance,
        'euclidean_distance': euclidean_distance
    }
    try:
        model_sia = keras.models.load_model("model_siamese_telapak.h5", custom_objects=custom_objects, compile=False)
    except Exception as e:
        st.warning(f"Model Siamese gagal dimuat: {e}")
        model_sia = None
    
    return model_clf, model_sia

# Load Database Profil JSON
try:
    with open("database_41_orang.json", "r") as f:
        DATABASE = json.load(f)
except FileNotFoundError:
    DATABASE = {}

CLASS_NAMES = [f"{i+1:03d}" for i in range(41)]

# ==========================================
# 3. FUNGSI PREPROCESSING
# ==========================================
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==========================================
# 4. TAMPILAN UTAMA (UI)
# ==========================================
def main():
    # Header
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1>🖐️ Sistem Keamanan Biometrik</h1>
        <p>Integrasi Klasifikasi MobileNetV2 & Verifikasi Siamese Network</p>
        <hr>
    </div>
    """, unsafe_allow_html=True)

    # Load Model
    model_clf, model_siamese = load_models()
    if not model_clf: 
        st.error("Model Utama tidak ditemukan. Harap cek file .h5")
        return

    # Layout Dua Kolom
    col_input, col_hasil = st.columns([1, 1.5], gap="large")

    # --- KOLOM KIRI: INPUT ---
    with col_input:
        st.subheader("1. Input Citra")
        uploaded_file = st.file_uploader("Upload Foto Telapak Tangan", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            image_display = Image.open(uploaded_file)
            st.image(image_display, caption="Preview Input", use_column_width=True)
            
            run_process = st.button("🔍 SCAN & VERIFIKASI", type="primary", use_container_width=True)

    # --- KOLOM KANAN: HASIL ---
    with col_hasil:
        st.subheader("2. Hasil Identifikasi")

        if uploaded_file and run_process:
            with st.spinner('Menganalisis pola biometrik...'):
                
                # A. KLASIFIKASI (Menebak ID)
                # MobileNet butuh 224x224
                input_clf = preprocess_image(image_display, (224, 224)) 
                preds = model_clf.predict(input_clf)
                
                idx = np.argmax(preds)
                conf_clf = float(np.max(preds) * 100)
                id_user = CLASS_NAMES[idx]
                
                # Ambil Data Profil dari JSON
                info = DATABASE.get(id_user, {'nama': 'Unknown', 'jabatan': '-', 'usia': '-'})

                # B. VERIFIKASI SIAMESE (Mengecek Kemiripan)
                jarak = -1.0
                status_verif = "UNVERIFIED"
                
                if model_siamese:
                    # Cari foto referensi di folder
                    path_ref = os.path.join("database_foto", f"{id_user}.jpg")
                    
                    if os.path.exists(path_ref):
                        img_ref = Image.open(path_ref)
                        
                        # Siamese butuh 128x128 (Sesuaikan dengan training Anda)
                        SIZE_SIA = (128, 128)
                        in_1 = preprocess_image(image_display, SIZE_SIA)
                        in_2 = preprocess_image(img_ref, SIZE_SIA)
                        
                        # PREDIKSI JARAK (Dual Input)
                        pred_sia = model_siamese.predict([in_1, in_2])
                        jarak = float(pred_sia[0][0])
                        
                        # LOGIKA THRESHOLD
                        THRESHOLD = 0.5
                        if jarak < THRESHOLD:
                            status_verif = "VERIFIED"
                        else:
                            status_verif = "REJECTED (Low Similarity)"
                    else:
                        st.warning(f"Foto referensi {id_user}.jpg tidak ditemukan. Verifikasi dilewati.")
                
                # C. MENAMPILKAN KARTU IDENTITAS
                if status_verif == "VERIFIED":
                    theme_color = "#27ae60" # Hijau
                    bg_theme = "#e9f7ef"
                    icon = "✅"
                else:
                    theme_color = "#c0392b" # Merah
                    bg_theme = "#fdedec"
                    icon = "⛔"

                # Kita simpan HTML ke variabel dulu agar rapi
                html_card = f"""
                <div style="
                    border: 2px solid {theme_color};
                    border-radius: 15px;
                    padding: 25px;
                    background-color: {bg_theme};
                    font-family: sans-serif;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid {theme_color}; padding-bottom: 10px; margin-bottom: 15px;">
                        <h2 style="color: {theme_color}; margin: 0;">{icon} {status_verif}</h2>
                        <h3 style="margin: 0; color: #555;">ID: {id_user}</h3>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                        <div>
                            <p style="margin:0; font-size:0.85em; color:#666;">NAMA LENGKAP</p>
                            <p style="margin:0; font-size:1.1em; font-weight:bold; color:#333;">{info['nama']}</p>
                        </div>
                        <div>
                            <p style="margin:0; font-size:0.85em; color:#666;">JABATAN</p>
                            <p style="margin:0; font-size:1.1em; font-weight:bold; color:#333;">{info['jabatan']}</p>
                        </div>
                         <div>
                            <p style="margin:0; font-size:0.85em; color:#666;">USIA</p>
                            <p style="margin:0; font-size:1.1em; font-weight:bold; color:#333;">{info['usia']} Tahun</p>
                        </div>
                        <div>
                            <p style="margin:0; font-size:0.85em; color:#666;">JARAK SIAMESE</p>
                            <p style="margin:0; font-size:1.2em; font-weight:bold; color:{theme_color};">{jarak:.4f}</p>
                        </div>
                    </div>
                    <br>
                    <small style="color: #777;"><i>Klasifikasi Confidence: {conf_clf:.2f}%</i></small>
                </div>
                """
                
                # EKSEKUSI RENDER HTML (Wajib unsafe_allow_html=True)
                st.markdown(html_card, unsafe_allow_html=True)
        elif not uploaded_file:
            # Tampilan Kosong
            st.info("Silakan upload gambar untuk memulai proses identifikasi.")

if __name__ == '__main__':
    main()
