import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import keras
import zipfile
import io

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
    initial_sidebar_state="expanded"
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
        st.sidebar.success("✅ Model Klasifikasi berhasil dimuat")
    except Exception as e:
        st.sidebar.error(f"❌ Gagal memuat Model Klasifikasi: {e}")
        model_clf = None

    # B. Load Model Siamese (Verifikasi)
    custom_objects = {
        'EuclideanDistance': EuclideanDistance,
        'euclidean_distance': euclidean_distance
    }
    try:
        model_sia = keras.models.load_model("model_siamese_telapak.h5", custom_objects=custom_objects, compile=False)
        st.sidebar.success("✅ Model Siamese berhasil dimuat")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Model Siamese gagal dimuat: {e}")
        model_sia = None
    
    return model_clf, model_sia

# Load Database Profil JSON
try:
    with open("database_41_orang.json", "r") as f:
        DATABASE = json.load(f)
except FileNotFoundError:
    DATABASE = {}
    st.sidebar.warning("⚠️ File database_41_orang.json tidak ditemukan")

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
# 3.5 FUNGSI UNTUK MEMBUAT ZIP SAMPLE
# ==========================================
def create_sample_zip():
    """Membuat ZIP berisi sample foto dari database_foto"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Ambil 10 sample foto pertama dari folder database_foto
        database_path = "database_foto"
        if os.path.exists(database_path):
            files = sorted(os.listdir(database_path))[:10]  # Ambil 10 file pertama
            for filename in files:
                file_path = os.path.join(database_path, filename)
                if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    zip_file.write(file_path, arcname=filename)
    
    zip_buffer.seek(0)
    return zip_buffer

# ==========================================
# 4. SIDEBAR - INFORMASI & DOWNLOAD
# ==========================================
with st.sidebar:
    st.markdown("### 📋 Informasi Sistem")
    st.markdown("---")
    
    st.markdown("""
    **Tentang Sistem:**
    - Model: MobileNetV2 + Siamese Network
    - Dataset: 41 Orang
    - Input Size: 224x224 (Klasifikasi)
    - Input Size: 128x128 (Verifikasi)
    - Threshold Verifikasi: 0.5
    """)
    
    st.markdown("---")
    st.markdown("### 🧪 Testing Dataset")
    
    st.info("""
    **⚠️ PENTING UNTUK PENGUJIAN:**
    
    Sistem ini hanya dapat mengenali **41 orang** yang sudah dilatih dalam model.
    
    Untuk testing, gunakan foto dari dataset di bawah ini.
    """)
    
    # Tombol Download Sample (dari folder lokal)
    st.markdown("#### 📥 Download Sample Foto")
    
    if os.path.exists("database_foto") and len(os.listdir("database_foto")) > 0:
        try:
            zip_data = create_sample_zip()
            st.download_button(
                label="📦 Download 10 Sample Foto",
                data=zip_data,
                file_name="sample_telapak_tangan.zip",
                mime="application/zip",
                help="Download 10 foto sample untuk testing sistem",
                use_container_width=True
            )
            st.success("✅ Sample tersedia untuk download")
        except Exception as e:
            st.error(f"❌ Error membuat ZIP: {e}")
    else:
        st.warning("⚠️ Folder database_foto tidak ditemukan atau kosong")
    
    st.markdown("---")
    
    # LINK GOOGLE DRIVE - GANTI DENGAN LINK ANDA!
    st.markdown("#### 🔗 Dataset Lengkap")
    
    # ===== GANTI LINK INI DENGAN LINK GOOGLE DRIVE ANDA =====
    DATASET_LINK = "https://drive.google.com/drive/folders/1RtvIMfPSNCQnYbwj5Te5VrPpidxFAbmR?usp=sharing"
    # =========================================================
    
    st.info("""
    **Dataset Lengkap berisi:**
    - 41 identitas (ID: 001-041)
    - Total ~1000+ foto training
    - Format: JPG/PNG
    """)
    
    st.link_button(
        label="📥 Buka Google Drive",
        url=DATASET_LINK,
        help="Download dataset lengkap untuk testing lebih dalam",
        type="primary",
        use_container_width=True
    )
    
    st.caption("⬆️ Klik tombol di atas untuk membuka Google Drive")
    
    # Daftar ID yang tersedia
    st.markdown("---")
    with st.expander("📜 Daftar ID Terlatih (41 Orang)"):
        cols = st.columns(3)
        for i, class_name in enumerate(CLASS_NAMES):
            with cols[i % 3]:
                st.text(f"• {class_name}")
    
    st.markdown("---")
    
    # Panduan Penggunaan
    st.markdown("#### 📖 Cara Penggunaan")
    st.markdown("""
    1. Download sample/dataset
    2. Extract file ZIP
    3. Upload salah satu foto
    4. Klik "SCAN & VERIFIKASI"
    5. Lihat hasil identifikasi
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        <p>Sistem Biometrik v1.0</p>
        <p>Developed with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. TAMPILAN UTAMA (UI)
# ==========================================
def main():
    # Header dengan emoji dan styling
    st.markdown('<p class="main-title">🖐️ Sistem Keamanan Biometrik</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Integrasi Klasifikasi MobileNetV2 & Verifikasi Siamese Network</p>', unsafe_allow_html=True)
    
    # Banner Peringatan
    st.warning("""
    ⚠️ **PERHATIAN UNTUK PENGUJI:** Sistem ini hanya dapat mengenali 41 identitas yang sudah dilatih. 
    Silakan download sample foto dari **sidebar kiri** untuk testing yang akurat.
    """)
    
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
            
            # Panduan untuk dosen
            with st.expander("💡 Panduan untuk Dosen/Penguji"):
                st.markdown("""
                **Untuk mendapatkan foto testing yang valid:**
                
                1. **Opsi 1: Download Sample (Cepat)**
                   - Buka sidebar di kiri
                   - Klik "📦 Download 10 Sample Foto"
                   - Extract ZIP dan upload salah satu foto
                
                2. **Opsi 2: Dataset Lengkap (Lengkap)**
                   - Klik tombol "📥 Buka Google Drive" di sidebar
                   - Download dataset lengkap (41 orang)
                   - Extract dan pilih foto untuk testing
                
                3. **Catatan Penting:**
                   - Foto random dari internet **TIDAK** akan dikenali
                   - Sistem hanya mengenali 41 orang tertentu
                   - ID yang valid: 001 - 041
                   
                4. **Testing Skenario:**
                   - **Positive Test:** Upload foto dari dataset → Harus VERIFIED
                   - **Negative Test:** Upload foto random → Akan REJECTED
                """)

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
                    st.write(f"**ID Terdeteksi:** {id_user}")
                    st.write(f"**Jarak Euclidean:** {jarak:.6f}")
                
        elif not uploaded_file:
            st.info("ℹ️ Menunggu input gambar...")
            st.markdown("""
            **Cara Penggunaan:**
            1. Download sample foto dari **sidebar kiri** 📥
            2. Upload salah satu foto sample di kolom kiri
            3. Klik tombol **SCAN & VERIFIKASI**
            4. Tunggu hasil analisis muncul di sini
            5. Sistem akan menampilkan identitas dan status verifikasi
            
            ---
            
            **⚠️ Catatan Penting:**
            - Gunakan foto dari 41 orang yang sudah dilatih
            - Foto random tidak akan dikenali dengan akurat
            - Download sample di sidebar untuk testing
            - Sistem menggunakan dua tahap: Klasifikasi + Verifikasi
            """)
        else:
            st.warning("⚠️ Klik tombol SCAN & VERIFIKASI untuk memproses")

if __name__ == '__main__':
    main()
