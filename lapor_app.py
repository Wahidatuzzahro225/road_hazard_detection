import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
import os
import torch

import torch.serialization
torch.load = lambda *args, **kwargs: torch.serialization.load(*args, **kwargs, weights_only=False)

from ultralytics import YOLO
model = YOLO("best.pt")

st.set_page_config(layout="wide")
st.title("Road Hazard Detection & Pelaporan Jalan Rusak")


# Buat folder untuk menyimpan laporan dan gambar hasil deteksi
os.makedirs("laporan", exist_ok=True)
os.makedirs("laporan/gambar", exist_ok=True)

# Sidebar menu
menu = st.sidebar.selectbox(
    "Pilih Mode",
    ["Upload Gambar", "Upload Video", "Realtime Camera"]
)

# ====================== UPLOAD IMAGE ======================
if menu == "Upload Gambar":
    uploaded_img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_img is not None:
        file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        results = model(image, conf=0.3)
        annotated = results[0].plot()

        st.image(annotated, channels="BGR", caption="Detection Result")

        # Form pelaporan
        st.header("Form Pelaporan Kerusakan Jalan")
        with st.form("laporan_form_cam"):
            nama = st.text_input("Nama")
            alamat = st.text_input("Alamat (Nama Jalan/Patokan)")
            
            # Ganti input angka dengan text input untuk Link Share Location
            location_link = st.text_input(
                "Link Share Location", 
                placeholder="Tempel link Google Maps di sini (Contoh: https://maps.app.goo.gl/...)"
            )
            
            deskripsi = st.text_area("Deskripsi Kerusakan")
            kategori = st.selectbox("Kategori", ["Pothole", "Speed Bump", "Patched Road", "Lainnya"])
            
            submit = st.form_submit_button("Kirim Laporan")

            if submit:
                # --- VALIDASI ---
                if not nama or not alamat or not location_link:
                    st.error("❌ Semua kolom (Nama, Alamat, Link Lokasi) wajib diisi!")
                elif "maps" not in location_link and "goo.gl" not in location_link:
                    st.error("❌ Link tidak valid! Pastikan menyalin dari Google Maps.")
                elif len(deskripsi) < 10:
                    st.warning("⚠️ Mohon berikan deskripsi yang lebih jelas (minimal 10 karakter).")
                else:
                    st.success(f"✅ Laporan '{kategori}' berhasil dikirim!")
                    st.balloons() # Efek perayaan kecil
                    st.info(f"Terima kasih {nama}, laporan Anda di {alamat} akan segera diproses.")

# ====================== UPLOAD VIDEO ======================
elif menu == "Upload Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        frame_window = st.image([])

        # Tombol untuk stop video jika perlu
        stop_button = st.button("Stop Video & Isi Laporan")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_button:
                break

            results = model(frame, conf=0.3)
            annotated = results[0].plot()
            frame_window.image(annotated, channels="BGR")

        cap.release()
        
        # Tambahkan pembatas visual
        st.divider()

        # Form pelaporan diletakkan di luar loop agar muncul setelah video selesai/distop
        st.header("Form Pelaporan Kerusakan Jalan")
        
        with st.form("laporan_form_video", clear_on_submit=True):
            nama = st.text_input("Nama Pelapor")
            alamat = st.text_input("Alamat (Nama Jalan/Patokan)")
            
            # Input Link Lokasi
            location_link = st.text_input(
                "Link Share Location", 
                placeholder="Tempel link Google Maps di sini..."
            )
            
            deskripsi = st.text_area("Deskripsi Kerusakan")
            kategori = st.selectbox("Kategori", ["Pothole", "Speed Bump", "Patched Road", "Lainnya"])
            
            submit = st.form_submit_button("Kirim Laporan")

            if submit:
                # --- VALIDASI ---
                if not nama or not alamat or not location_link:
                    st.error("❌ Semua kolom (Nama, Alamat, Link Lokasi) wajib diisi!")
                elif "maps" not in location_link and "goo.gl" not in location_link:
                    st.error("❌ Link tidak valid! Pastikan menyalin dari Google Maps.")
                elif len(deskripsi) < 10:
                    st.warning("⚠️ Mohon berikan deskripsi yang lebih jelas (minimal 10 karakter).")
                else:
                    st.success(f"✅ Laporan '{kategori}' berhasil dikirim!")
                    st.balloons() # Efek perayaan kecil
                    st.info(f"Terima kasih {nama}, laporan Anda di {alamat} akan segera diproses.")

# ====================== REALTIME CAMERA ======================
elif menu == "Realtime Camera":
    
    # Inisialisasi session state untuk menyimpan frame terakhir
    if 'last_frame' not in st.session_state:
        st.session_state['last_frame'] = None

    run = st.checkbox("Aktifkan Kamera")
    frame_window = st.image([])

    # Bagian Kamera
    if run:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Gagal mengakses kamera.")
                break

            # Proses deteksi
            results = model(frame, conf=0.3)
            annotated = results[0].plot()
            
            # SIMPAN frame yang sudah ada kotaknya ke session state agar bisa dipakai Form
            st.session_state['last_frame'] = annotated
            
            # Tampilkan ke layar
            frame_window.image(annotated, channels="BGR")

            # Cek jika checkbox dimatikan saat loop berjalan
            if not run:
                break
        cap.release()
    else:
        st.info("Centang 'Aktifkan Kamera' untuk mulai deteksi.")

    # ================= Form Pelaporan =================
    st.divider()
    st.subheader("📝 Isi Form Laporan Berdasarkan Hasil Kamera")
    
    with st.form("laporan_form_cam", clear_on_submit=True):            
        nama = st.text_input("Nama Pelapor")
        alamat = st.text_input("Alamat (Nama Jalan/Patokan)")
        location_link = st.text_input("Link Share Location", placeholder="Tempel link Google Maps...")
        deskripsi = st.text_area("Deskripsi Kerusakan")
        kategori = st.selectbox("Kategori", ["Pothole", "Speed Bump", "Patched Road", "Lainnya"])
        
        submit = st.form_submit_button("Kirim Laporan")

        if submit:
            # 1. VALIDASI DATA FORM
            if not nama or not alamat or not location_link:
                st.error("❌ Semua kolom wajib diisi!")
            elif "maps" not in location_link and "goo.gl" not in location_link:
                st.error("❌ Link Google Maps tidak valid!")
            
            # 2. VALIDASI APAKAH SUDAH ADA GAMBAR YANG TERCAPTURE
            elif st.session_state['last_frame'] is None:
                st.warning("⚠️ Belum ada gambar yang tertangkap kamera. Jalankan kamera terlebih dahulu.")
            
            else:
                # BUAT FOLDER JIKA BELUM ADA
                os.makedirs("laporan/gambar", exist_ok=True)
                
                # NAMA FILE UNIK
                import time
                timestamp = int(time.time())
                img_name = f"{nama}_{kategori}_{timestamp}.jpg"
                img_path = os.path.join("laporan/gambar", img_name)
                
                # SIMPAN GAMBAR DARI SESSION STATE KE FOLDER (DATABASE FILE)
                cv2.imwrite(img_path, st.session_state['last_frame'])

                # SIMPAN DATA KE CSV (DATABASE SEDERHANA)
                # Di sini kamu bisa ganti pakai SQL kalau mau
                import pandas as pd
                new_data = {
                    "Waktu": [time.ctime()],
                    "Nama": [nama],
                    "Alamat": [alamat],
                    "Link_Lokasi": [location_link],
                    "Kategori": [kategori],
                    "Path_Gambar": [img_path]
                }
                df = pd.DataFrame(new_data)
                
                # Simpan/Append ke CSV
                csv_path = "laporan/database_laporan.csv"
                if not os.path.isfile(csv_path):
                    df.to_csv(csv_path, index=False)
                else:
                    df.to_csv(csv_path, mode='a', header=False, index=False)

                st.success(f"✅ Laporan Berhasil Dikirim!")
                st.image(img_path, caption="Gambar yang Dilaporkan", width=300)
                st.info(f"Data tersimpan di database dan gambar di: {img_path}")
