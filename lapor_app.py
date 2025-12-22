import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
import os
import time
from datetime import datetime

# KONFIGURASI HALAMAN HARUS DI PALING ATAS
st.set_page_config(layout="wide", page_title="Road Hazard Detection")

# Buat folder untuk simpan laporan
os.makedirs("laporan", exist_ok=True)
os.makedirs("laporan/gambar", exist_ok=True)

# Load model YOLO dengan error handling
@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        return YOLO("best.pt")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Load model di sini
with st.spinner("Loading AI model..."):
    model = load_model()

# Judul aplikasi
st.title("🛣️ Road Hazard Detection & Pelaporan Jalan Rusak")

# Sidebar menu
menu = st.sidebar.selectbox(
    "🎯 Pilih Mode",
    ["Upload Gambar", "Upload Video", "Realtime Camera"]
)

# ====================== UPLOAD IMAGE ======================
if menu == "Upload Gambar":
    st.subheader("📷 Upload Gambar Jalan")
    uploaded_img = st.file_uploader("Pilih gambar jalan rusak", type=["jpg", "png", "jpeg"])

    if uploaded_img is not None:
        file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Mendeteksi kerusakan jalan..."):
            results = model(image, conf=0.3)
            annotated = results[0].plot()

        st.image(annotated, channels="BGR", caption="Hasil Deteksi", use_container_width=True)

        # Form pelaporan
        st.divider()
        st.header("📝 Form Pelaporan Kerusakan Jalan")
        
        with st.form("laporan_form_img"):
            col1, col2 = st.columns(2)
            
            with col1:
                nama = st.text_input("Nama Lengkap *")
                alamat = st.text_input("Alamat/Lokasi *")
            
            with col2:
                location_link = st.text_input(
                    "Link Google Maps *", 
                    placeholder="https://maps.app.goo.gl/..."
                )
                kategori = st.selectbox("Kategori Kerusakan *", 
                    ["Pothole", "Speed Bump", "Patched Road", "Lainnya"])
            
            deskripsi = st.text_area("Deskripsi Kerusakan *", 
                placeholder="Jelaskan kondisi kerusakan jalan...")
            
            submit = st.form_submit_button("✅ Kirim Laporan", use_container_width=True)

            if submit:
                if not nama or not alamat or not location_link:
                    st.error("❌ Semua kolom bertanda * wajib diisi!")
                elif "maps" not in location_link.lower() and "goo.gl" not in location_link.lower():
                    st.error("❌ Link tidak valid! Pastikan menyalin dari Google Maps.")
                elif len(deskripsi) < 10:
                    st.warning("⚠️ Deskripsi terlalu singkat (minimal 10 karakter).")
                else:
                    # Simpan gambar
                    timestamp = int(time.time())
                    img_name = f"{nama.replace(' ', '_')}_{kategori}_{timestamp}.jpg"
                    img_path = os.path.join("laporan/gambar", img_name)
                    cv2.imwrite(img_path, annotated)
                    
                    # Simpan ke CSV
                    new_data = {
                        "Waktu": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                        "Nama": [nama],
                        "Alamat": [alamat],
                        "Link_Lokasi": [location_link],
                        "Deskripsi": [deskripsi],
                        "Kategori": [kategori],
                        "Path_Gambar": [img_path]
                    }
                    df = pd.DataFrame(new_data)
                    
                    csv_path = "laporan/database_laporan.csv"
                    if not os.path.isfile(csv_path):
                        df.to_csv(csv_path, index=False)
                    else:
                        df.to_csv(csv_path, mode='a', header=False, index=False)
                    
                    st.success(f"✅ Laporan '{kategori}' berhasil dikirim!")
                    st.balloons()
                    st.info(f"📧 Terima kasih {nama}! Laporan Anda akan segera ditindaklanjuti.")

# ====================== UPLOAD VIDEO ======================
elif menu == "Upload Video":
    st.subheader("🎥 Upload Video Jalan")
    uploaded_video = st.file_uploader("Pilih video jalan rusak", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        frame_window = st.empty()
        
        col1, col2 = st.columns([1, 4])
        with col1:
            stop_button = st.button("⏹️ Stop Video")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_button:
                break

            # Process setiap 3 frame untuk performa
            if frame_count % 3 == 0:
                results = model(frame, conf=0.3)
                annotated = results[0].plot()
                frame_window.image(annotated, channels="BGR", use_container_width=True)
            
            frame_count += 1

        cap.release()
        
        st.divider()
        st.header("📝 Form Pelaporan")
        
        with st.form("laporan_form_video", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                nama = st.text_input("Nama Lengkap *")
                alamat = st.text_input("Alamat/Lokasi *")
            
            with col2:
                location_link = st.text_input("Link Google Maps *", 
                    placeholder="https://maps.app.goo.gl/...")
                kategori = st.selectbox("Kategori Kerusakan *", 
                    ["Pothole", "Speed Bump", "Patched Road", "Lainnya"])
            
            deskripsi = st.text_area("Deskripsi Kerusakan *")
            
            submit = st.form_submit_button("✅ Kirim Laporan", use_container_width=True)

            if submit:
                if not nama or not alamat or not location_link:
                    st.error("❌ Semua kolom bertanda * wajib diisi!")
                elif "maps" not in location_link.lower() and "goo.gl" not in location_link.lower():
                    st.error("❌ Link Google Maps tidak valid!")
                elif len(deskripsi) < 10:
                    st.warning("⚠️ Deskripsi terlalu singkat.")
                else:
                    # Simpan data ke CSV
                    new_data = {
                        "Waktu": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                        "Nama": [nama],
                        "Alamat": [alamat],
                        "Link_Lokasi": [location_link],
                        "Deskripsi": [deskripsi],
                        "Kategori": [kategori],
                        "Path_Gambar": ["video_upload"]
                    }
                    df = pd.DataFrame(new_data)
                    
                    csv_path = "laporan/database_laporan.csv"
                    if not os.path.isfile(csv_path):
                        df.to_csv(csv_path, index=False)
                    else:
                        df.to_csv(csv_path, mode='a', header=False, index=False)
                    
                    st.success(f"✅ Laporan '{kategori}' berhasil dikirim!")
                    st.balloons()

# ====================== REALTIME CAMERA ======================
elif menu == "Realtime Camera":
    st.warning("⚠️ Fitur kamera realtime tidak tersedia di Streamlit Cloud. Gunakan mode Upload Gambar atau Video.")
    
    st.divider()
    st.info("""
    ### 💡 Alternatif:
    1. **Ambil foto/video** dengan kamera HP
    2. **Upload** menggunakan menu di sidebar
    3. **Isi form** pelaporan
    """)
