<<<<<<< HEAD
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
=======
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
>>>>>>> 5a89ef314d09fc40f633f6fc05431870735d8187
import os
import keras

# ==========================================
<<<<<<< HEAD
# 1. DEFINISI CUSTOM OBJECTS
# ==========================================

# Definisi Function (Penyebab error "function not found")
=======
# 1. DEFINISI CUSTOM LAYER 
# ==========================================
>>>>>>> 5a89ef314d09fc40f633f6fc05431870735d8187
@keras.saving.register_keras_serializable()
def euclidean_distance(vectors):
    x, y = vectors
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

<<<<<<< HEAD
# Definisi Class (Jaga-jaga)
=======
>>>>>>> 5a89ef314d09fc40f633f6fc05431870735d8187
@keras.saving.register_keras_serializable()
class EuclideanDistance(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        x, y = inputs
        return tf.math.reduce_sum(tf.math.square(x - y), axis=-1, keepdims=True)

<<<<<<< HEAD
app = Flask(__name__)
CORS(app)

# ==========================================
# 2. LOAD MODEL & DATABASE
# ==========================================
print("\n" + "="*50)
print("--- STARTING BACKEND SYSTEM ---")

# Database JSON
try:
    with open("database_41_orang.json", "r") as f:
        DATABASE = json.load(f)
except:
    DATABASE = {}
CLASS_NAMES = [f"{i+1:03d}" for i in range(41)]

# Model Klasifikasi
try:
    model_clf = keras.models.load_model("model_klasifikasi_telapak.h5", compile=False)
    print("✅ Model Klasifikasi: LOADED")
except Exception as e:
    print(f"❌ Model Klasifikasi ERROR: {e}")
    model_clf = None

# Model Verifikasi (Siamese)
custom_objects = {
    'EuclideanDistance': EuclideanDistance,
    'euclidean_distance': euclidean_distance
}
try:
    model_siamese = keras.models.load_model("model_siamese_telapak.h5", custom_objects=custom_objects, compile=False)
    print("✅ Model Siamese: LOADED (Dual Input Mode)")
except Exception as e:
    print(f"⚠️ Model Siamese ERROR: {e}")
    model_siamese = None

FOLDER_REF = "database_foto"

# ==========================================
# 3. PREPROCESSING
=======
# ==========================================
# 2. LOAD MODEL
# ==========================================
@st.cache_resource
def load_models():
    # Load Klasifikasi
    try:
        model_clf = keras.models.load_model("model_klasifikasi_telapak.h5", compile=False)
    except:
        model_clf = None

    # Load Siamese
    custom_objects = {'EuclideanDistance': EuclideanDistance, 'euclidean_distance': euclidean_distance}
    try:
        model_ver = keras.models.load_model("model_siamese_telapak.h5", custom_objects=custom_objects, compile=False)
    except:
        model_ver = None
    
    return model_clf, model_ver

# Load Database Nama Dummy
CLASS_NAMES = [f"{i+1:03d}" for i in range(41)]
DATABASE = {
    '001': {'nama': 'Andi Pratama', 'jabatan': 'Mahasiswa', 'usia': 21},
    '002': {'nama': 'Budi Santoso', 'jabatan': 'Dosen', 'usia': 45},
    '005': {'nama': 'Eko Kurniawan', 'jabatan': 'Laboran', 'usia': 35},
    '041': {'nama': 'Nurul Putri', 'jabatan': 'Mahasiswa', 'usia': 22}
}

# ==========================================
# 3. FUNGSI UTILITY
>>>>>>> 5a89ef314d09fc40f633f6fc05431870735d8187
# ==========================================
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==========================================
<<<<<<< HEAD
# 4. ROUTE PREDIKSI (FIXED DUAL INPUT)
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    try:
        # --- 1. GAMBAR INPUT (Dari React) ---
        file = request.files['file']
        img_input_pil = Image.open(io.BytesIO(file.read()))
        
        print("\n📸 New Request Received")

        # --- 2. KLASIFIKASI (Tebak ID) ---
        if model_clf:
            input_clf = preprocess_image(img_input_pil, (224, 224))
            preds = model_clf.predict(input_clf, verbose=0)
            idx = np.argmax(preds)
            conf_clf = float(np.max(preds) * 100)
            id_user = CLASS_NAMES[idx]
            print(f"   Step 1 (Klasifikasi): ID-{id_user} | Conf: {conf_clf:.2f}%")
        else:
            id_user = "001"
            conf_clf = 0.0

        user_data = DATABASE.get(id_user, {"nama": "Unknown", "jabatan": "-", "usia": "-"})

        # --- 3. VERIFIKASI (Siamese Dual Input) ---
        jarak = -1.0
        status = "UNVERIFIED"

        if model_siamese:
            # Cari GAMBAR REFERENSI di folder server
            path_ref = os.path.join(FOLDER_REF, f"{id_user}.jpg")
            
            if os.path.exists(path_ref):
                print(f"   Step 2 (Verifikasi): Membandingkan Input vs {id_user}.jpg")
                img_ref_pil = Image.open(path_ref)

                # Siapkan 2 Gambar
                # Ubah ukuran sesuai input shape Siamese Anda (misal 128x128)
                SIZE = (128, 128) 
                in_1 = preprocess_image(img_input_pil, SIZE) # Gambar Upload
                in_2 = preprocess_image(img_ref_pil, SIZE)   # Gambar Database

                # === PERBAIKAN PENTING DI SINI ===
                # Kirim List [in_1, in_2] ke model
                prediksi_siamese = model_siamese.predict([in_1, in_2], verbose=0)
                
                # Output model adalah [[jarak]], ambil angkanya
                jarak = float(prediksi_siamese[0][0])
                print(f"   >>> Jarak Euclidean: {jarak:.4f}")

                # Threshold Logic
                THRESHOLD = 0.5
                if jarak < THRESHOLD:
                    status = "VERIFIED"
                else:
                    status = "UNVERIFIED"
            else:
                print(f"⚠️ Foto referensi {id_user}.jpg tidak ada. Verifikasi dilewati.")
        
        return jsonify({
            "data": user_data,
            "confidence": conf_clf,
            "status_verifikasi": status,
            "meta": {
                "klasifikasi_confidence": conf_clf,
                "verifikasi_distance": jarak,
                "pesan": "Sukses"
            }
        })

    except Exception as e:
        print(f"🔥 FATAL ERROR: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists(FOLDER_REF):
        os.makedirs(FOLDER_REF)
    app.run(port=5000, debug=True)
=======
# 4. TAMPILAN UTAMA (UI)
# ==========================================
def main():
    st.set_page_config(page_title="UAS Biometrik", page_icon="🖐️", layout="wide")
    
    st.markdown("<h1 style='text-align: center;'>🖐️ Sistem Keamanan Biometrik</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Klasifikasi (MobileNetV2) & Verifikasi (Siamese Network)</p>", unsafe_allow_html=True)
    st.write("---")

    # Load Model
    model_clf, model_siamese = load_models()

    if not model_clf or not model_siamese:
        st.error("Model gagal dimuat. Pastikan file .h5 sudah diupload ke GitHub.")
        return

    col1, col2 = st.columns([1, 1.5])

    # --- INPUT ---
    with col1:
        st.subheader("1. Input Gambar")
        uploaded_file = st.file_uploader("Upload Telapak Tangan", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Preview", use_column_width=True)
            run_btn = st.button("🔍 SCAN & VERIFIKASI", type="primary", use_container_width=True)

    # --- HASIL ---
    with col2:
        st.subheader("2. Hasil Analisis")
        
        if uploaded_file and run_btn:
            with st.spinner('Memproses kecerdasan buatan...'):
                # 1. KLASIFIKASI
                input_clf = preprocess_image(image, (224, 224))
                preds = model_clf.predict(input_clf)
                idx = np.argmax(preds)
                conf = np.max(preds) * 100
                id_user = CLASS_NAMES[idx]
                
                # Ambil Info User
                info = DATABASE.get(id_user, {'nama': 'Unknown User', 'jabatan': 'Tamu', 'usia': '-'})

                # 2. VERIFIKASI (SIAMESE)
                jarak = -1.0
                path_ref = os.path.join("database_foto", f"{id_user}.jpg")
                
                # Cek apakah foto referensi ada di GitHub/Folder
                if os.path.exists(path_ref):
                    img_ref = Image.open(path_ref)
                    
                    # Dual Input Preprocessing
                    SIZE_SIA = (128, 128) # Sesuaikan dgn model siamese Anda
                    in_1 = preprocess_image(image, SIZE_SIA)
                    in_2 = preprocess_image(img_ref, SIZE_SIA)
                    
                    # PREDIKSI SIAMESE (DUAL INPUT)
                    jarak = float(model_siamese.predict([in_1, in_2])[0][0])
                else:
                    st.warning(f"Foto referensi {id_user}.jpg tidak ditemukan di server.")

                # 3. LOGIKA FINAL
                THRESHOLD = 0.5
                is_verified = jarak != -1.0 and jarak < THRESHOLD

                # 4. TAMPILAN KARTU
                if is_verified:
                    bg_color = "#e6fffa"
                    border_color = "#2ecc71"
                    status = "✅ VERIFIED ACCESS"
                else:
                    bg_color = "#fff5f5"
                    border_color = "#e74c3c"
                    status = "⛔ ACCESS DENIED"

                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; border-left: 10px solid {border_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h2 style="color: {border_color}; margin:0;">{status}</h2>
                    <hr>
                    <p><strong>ID:</strong> {id_user}</p>
                    <p><strong>NAMA:</strong> {info['nama']}</p>
                    <p><strong>JABATAN:</strong> {info['jabatan']}</p>
                    <br>
                    <p><small>Jarak Siamese: {jarak:.4f} (Threshold: {THRESHOLD})</small></p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
>>>>>>> 5a89ef314d09fc40f633f6fc05431870735d8187
