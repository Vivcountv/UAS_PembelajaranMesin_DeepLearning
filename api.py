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
st.set_page_config(
    page_title="Sistem Keamanan Biometrik", 
    page_icon="🖐️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk styling tambahan
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #155a8a;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    # A. Load Model Klasifikasi
    try:
        model_clf = keras.models.load_model("model_klasifikasi_telapak.h5", compile=False)
        st.success("✅ Model Klasifikasi berhasil dimuat")
    except Exception as e:
        st.error(f"❌ Gagal memuat Model Klasifikasi: {e}")
        model_clf = None

    # B. Load Model Siamese (Verifikasi)
    custom_objects = {
        'EuclideanDistance': EuclideanDistance,
        'euclidean_distance': euclidean_distance
    }
    try:
        model_sia = keras.models.load_model("model_siamese_telapak.h5", custom_objects=custom_objects, compile=False)
        st.success("✅ Model Siamese berhasil dimuat")
    except Exception as e:
        st.warning(f"⚠️ Model Siamese gagal dimuat: {e}")
        model_sia = None
    
    return model_clf, model_sia

# Load Database Profil JSON
try:
    with open("database_41_orang.json", "r") as f:
        DATABASE = json.load(f)
except FileNotFoundError:
    DATABASE = {}
    st.warning("⚠️ File database_41_orang.json tidak ditemukan")

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
    # Header dengan emoji dan styling
    st.markdown('<p class="main-title">🖐️ Sistem Keamanan Biometrik</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Integrasi Klasifikasi MobileNetV2 & Verifikasi Siamese Network</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Load Model
    model_clf, model_siamese = load_models()
    
    if not model_clf: 
        st.error("⛔ Model Utama tidak ditemukan. Harap cek file .h5")
        st.stop()

    # Layout Dua Kolom
    col_input, col_hasil = st.columns([1, 1.8], gap="large")

    # --- KOLOM KIRI: INPUT ---
    with col_input:
        st.markdown("### 📸 Input Citra")
        uploaded_file = st.file_uploader(
            "Upload Foto Telapak Tangan", 
            type=["jpg", "png", "jpeg"],
            help="Format: JPG, PNG, JPEG"
        )
        
        if uploaded_file:
            image_display = Image.open(uploaded_file)
            st.image(image_display, caption="Preview Gambar", use_column_width=True)
            st.markdown("")  # Spacing
            run_process = st.button("🔍 SCAN & VERIFIKASI", type="primary")
        else:
            st.info("👆 Silakan upload gambar telapak tangan untuk memulai")

    # --- KOLOM KANAN: HASIL ---
    with col_hasil:
        st.markdown("### 📊 Hasil Identifikasi")

        if uploaded_file and run_process:
            with st.spinner('🔄 Menganalisis pola biometrik...'):
                
                # A. KLASIFIKASI (Menebak ID)
                input_clf = preprocess_image(image_display, (224, 224)) 
                preds = model_clf.predict(input_clf, verbose=0)
                
                idx = np.argmax(preds)
                conf_clf = float(np.max(preds) * 100)
                id_user = CLASS_NAMES[idx]
                
                # Ambil Data Profil dari JSON
                info = DATABASE.get(id_user, {'nama': 'Unknown', 'jabatan': '-', 'usia': '-'})

                # B. VERIFIKASI SIAMESE (Mengecek Kemiripan)
                jarak = -1.0
                status_verif = "UNVERIFIED"
                
                if model_siamese:
                    path_ref = os.path.join("database_foto", f"{id_user}.jpg")
                    
                    if os.path.exists(path_ref):
                        img_ref = Image.open(path_ref)
                        
                        SIZE_SIA = (128, 128)
                        in_1 = preprocess_image(image_display, SIZE_SIA)
                        in_2 = preprocess_image(img_ref, SIZE_SIA)
                        
                        pred_sia = model_siamese.predict([in_1, in_2], verbose=0)
                        jarak = float(pred_sia[0][0])
                        
                        THRESHOLD = 0.5
                        if jarak < THRESHOLD:
                            status_verif = "VERIFIED"
                        else:
                            status_verif = "REJECTED"
                    else:
                        st.warning(f"⚠️ Foto referensi {id_user}.jpg tidak ditemukan")
                
                # C. TAMPILAN HASIL DENGAN STREAMLIT NATIVE
                st.markdown("")  # Spacing
                
                # Status Badge
                if status_verif == "VERIFIED":
                    st.success(f"### ✅ AKSES DITERIMA - {status_verif}")
                    status_color = "green"
                else:
                    st.error(f"### ⛔ AKSES DITOLAK - {status_verif}")
                    status_color = "red"
                
                st.markdown("---")
                
                # Informasi dalam Container
                with st.container():
                    # Baris 1: ID dan Confidence
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric(
                            label="🆔 ID Pengguna",
                            value=id_user,
                            delta=None
                        )
                    with metric_col2:
                        st.metric(
                            label="📊 Confidence Score",
                            value=f"{conf_clf:.2f}%",
                            delta=f"{conf_clf - 50:.1f}%" if conf_clf > 50 else None
                        )
                    
                    st.markdown("")
                    
                    # Baris 2: Nama dan Jabatan
                    info_col1, info_col2 = st.columns(2)
                    with info_col1:
                        st.markdown(f"""
                        **👤 NAMA LENGKAP**  
                        `{info['nama']}`
                        """)
                    with info_col2:
                        st.markdown(f"""
                        **💼 JABATAN**  
                        `{info['jabatan']}`
                        """)
                    
                    st.markdown("")
                    
                    # Baris 3: Usia dan Jarak Siamese
                    detail_col1, detail_col2 = st.columns(2)
                    with detail_col1:
                        st.markdown(f"""
                        **🎂 USIA**  
                        `{info['usia']} Tahun`
                        """)
                    with detail_col2:
                        jarak_color = "🟢" if jarak < 0.5 else "🔴"
                        st.markdown(f"""
                        **📏 JARAK SIAMESE**  
                        {jarak_color} `{jarak:.4f}`
                        """)
                
                st.markdown("---")
                
                # Progress Bar untuk visualisasi confidence
                st.markdown("**Tingkat Kepercayaan Klasifikasi:**")
                st.progress(min(conf_clf / 100, 1.0))
                
                # Expander untuk detail teknis
                with st.expander("🔬 Detail Teknis"):
                    st.write(f"**Model Klasifikasi:** MobileNetV2")
                    st.write(f"**Model Verifikasi:** Siamese Network")
                    st.write(f"**Threshold Siamese:** 0.5")
                    st.write(f"**Prediksi Index:** {idx}")
                    st.write(f"**Raw Confidence:** {conf_clf:.4f}%")
                    st.write(f"**Status Verifikasi:** {status_verif}")
                
        elif not uploaded_file:
            st.info("ℹ️ Menunggu input gambar...")
            st.markdown("""
            **Cara Penggunaan:**
            1. Upload foto telapak tangan di kolom sebelah kiri
            2. Klik tombol **SCAN & VERIFIKASI**
            3. Tunggu hasil analisis muncul di sini
            4. Sistem akan menampilkan identitas dan status verifikasi
            """)
        else:
            st.warning("⚠️ Klik tombol SCAN & VERIFIKASI untuk memproses")

if __name__ == '__main__':
    main()
