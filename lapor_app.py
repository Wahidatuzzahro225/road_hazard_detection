import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
import os
import time
from datetime import datetime
import torch

# PATCH torch.load SEBELUM import YOLO
_original_torch_load = torch.load

def patched_torch_load(f, *args, **kwargs):
    # Force weights_only=False untuk ultralytics
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)

torch.load = patched_torch_load

# Import ultralytics SETELAH patch
from ultralytics import YOLO

# KONFIGURASI HALAMAN HARUS DI PALING ATAS
st.set_page_config(layout="wide", page_title="Road Hazard Detection")

# Buat folder untuk simpan laporan
os.makedirs("laporan", exist_ok=True)
os.makedirs("laporan/gambar", exist_ok=True)

# Load model YOLO dengan error handling
@st.cache_resource
def load_model():
    try:
        return YOLO("best.pt")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Load model di sini
with st.spinner("Loading AI model..."):
    model = load_model()

# Judul aplikasi
st.title("Road Hazard Detection & Pelaporan Jalan Rusak")

# Sidebar menu
menu = st.sidebar.selectbox(
    "Pilih Mode",
    ["Upload Gambar", "Upload Video", "Realtime Camera"]
)

# ====================== UPLOAD IMAGE ======================
if menu == "Upload Gambar":
    st.subheader("Upload Gambar Jalan")
    uploaded_img = st.file_uploader("Pilih gambar jalan rusak", type=["jpg", "png", "jpeg"])

    if uploaded_img is not None:
        file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Mendeteksi kerusakan jalan..."):
            results = model(image, conf=0.3)
            annotated = results[0].plot()

        st.image(annotated, channels="BGR", caption="Hasil Deteksi", width=600)

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
    st.subheader("Upload Video Jalan")
    uploaded_video = st.file_uploader("Pilih video jalan rusak", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        
        # Info video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30
        
        st.info(f"📊 Total frames: {total_frames} | FPS: {fps:.1f}")
        
        # Pilihan mode
        mode = st.radio(
            "Pilih Mode Proses:",
            ["Real-time (Lambat tapi smooth)", "Fast (Cepat, skip beberapa frame)"],
            horizontal=True
        )
        
        # Buat placeholder
        frame_window = st.empty()
        progress_bar = st.progress(0)
        
        # Tombol start
        if st.button("▶️ Mulai Proses Video"):
            frame_count = 0
            skip_frames = 1 if mode.startswith("Real-time") else 3
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frame jika mode Fast
                if frame_count % skip_frames == 0:
                    results = model(frame, conf=0.3)
                    annotated = results[0].plot()
                    frame_window.image(annotated, channels="BGR", width=600)
                
                frame_count += 1
                
                # Update progress
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            progress_bar.progress(1.0)
            st.success(f"✅ Video selesai! Total {frame_count} frames diproses.")

        # Tambahkan pembatas visual
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
    st.subheader("Realtime Camera Detection")
    
    # Deteksi apakah di cloud atau lokal
    is_cloud = os.environ.get('STREAMLIT_SHARING_MODE') or os.environ.get('IS_DOCKER')
    
    if is_cloud:
        st.warning("""
        ⚠️ **Kamera Realtime tidak tersedia di Streamlit Cloud**
        
        Streamlit Cloud tidak bisa akses kamera perangkat karena limitasi keamanan.
        
        ### 💡 Solusi:
        1. **Gunakan mode "Upload Gambar"** - Ambil foto dengan HP lalu upload
        2. **Jalankan di lokal** - Download kode dan jalankan di komputer dengan webcam
        
        ### 🖥️ Cara Jalankan di Lokal:
        ```bash
        # Clone repository
        git clone https://github.com/[username]/road_hazard_detection.git
        cd road_hazard_detection
        
        # Install dependencies
        pip install -r requirements.txt
        
        # Jalankan aplikasi
        streamlit run lapor_app.py
        ```
        """)
    else:
        # KODE KAMERA UNTUK LOKAL
        if 'last_frame' not in st.session_state:
            st.session_state['last_frame'] = None
        
        run = st.checkbox("Aktifkan Kamera")
        frame_window = st.empty()
        
        if run:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Tidak dapat mengakses kamera. Baca solusi di lapor_app.py untuk menjalankan di streamlit lokal. ")
            else:
                st.info("✅ Kamera aktif. Arahkan ke jalan rusak untuk deteksi otomatis.")
                
                while run and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Gagal membaca frame dari kamera.")
                        break
                    
                    # Deteksi dengan YOLO
                    results = model(frame, conf=0.3)
                    annotated = results[0].plot()
                    
                    # Simpan frame untuk form
                    st.session_state['last_frame'] = annotated
                    
                    # Tampilkan
                    frame_window.image(annotated, channels="BGR", width=600)
                    
                    # Break jika checkbox dimatikan
                    if not st.session_state.get('camera_running', True):
                        break
                
                cap.release()
        else:
            st.info("ℹ️ Centang checkbox di atas untuk mengaktifkan kamera.")
    
    # FORM PELAPORAN (Bisa dipakai di cloud maupun lokal)
    st.divider()
    st.subheader("📝 Isi Form Laporan")
    
    with st.form("laporan_form_cam", clear_on_submit=True):
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
            elif st.session_state.get('last_frame') is None and not is_cloud:
                st.warning("⚠️ Belum ada gambar dari kamera. Aktifkan kamera terlebih dahulu.")
            else:
                # Simpan gambar jika ada
                img_path = "camera_capture"
                if st.session_state.get('last_frame') is not None:
                    timestamp = int(time.time())
                    img_name = f"{nama.replace(' ', '_')}_{kategori}_{timestamp}.jpg"
                    img_path = os.path.join("laporan/gambar", img_name)
                    cv2.imwrite(img_path, st.session_state['last_frame'])
                
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
                
                if img_path != "camera_capture":
                    st.image(img_path, caption="Gambar yang Dilaporkan", width=300)
