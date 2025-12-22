import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

st.set_page_config(layout="wide")
st.title("Road Hazard Detection (YOLOv8)")

model = YOLO("best.pt")

menu = st.sidebar.selectbox(
    "Pilih Mode",
    ["Upload Gambar", "Upload Video", "Realtime Camera"]
)

#  UPLOAD IMAGE
if menu == "Upload Gambar":
    uploaded_img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_img is not None:
        file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        results = model(image, conf=0.3)
        annotated = results[0].plot()

        st.image(annotated, channels="BGR", caption="Detection Result")

# 🎥 UPLOAD VIDEO
elif menu == "Upload Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        frame_window = st.image([])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.3)
            annotated = results[0].plot()
            frame_window.image(annotated, channels="BGR")

        cap.release()

# 📷 REALTIME CAMERA
elif menu == "Realtime Camera":
    run = st.checkbox("Run Camera")
    frame_window = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera tidak bisa diakses")
            break

        results = model(frame, conf=0.3)
        annotated = results[0].plot()
        frame_window.image(annotated, channels="BGR")

    cap.release()
