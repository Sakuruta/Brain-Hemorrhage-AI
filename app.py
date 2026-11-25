import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp
from src.metrics import ekstrak_analisis_pendarahan
from src.visualizer import gambar_hasil_deteksi

st.set_page_config(page_title="Deteksi Pendarahan Otak CT Scan", layout="wide")
st.title("Deteksi & Estimasi Dimensi Pendarahan Otak (Non-Contrast CT)")

IMAGE_SIZE = 256
NUM_CLASSES = 6  # Pastikan sama dengan NUM_CLASSES di train.py
MODEL_PATH = './models/best_unet.pt'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = smp.Unet(encoder_name="mobilenet_v2", in_channels=3, classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

uploaded_file = st.file_uploader("Unggah irisan CT Scan Otak (.jpg / .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    model = load_model()
    image = Image.open(uploaded_file).convert("RGB")
    raw_rgb = np.array(image)
    resized_rgb = cv2.resize(raw_rgb, (IMAGE_SIZE, IMAGE_SIZE))

    # Preprocessing
    inp = resized_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = np.transpose((inp - mean) / std, (2, 0, 1))
    tensor_inp = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor_inp)
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    hasil = ekstrak_analisis_pendarahan(pred_mask, img_size=IMAGE_SIZE)
    vis_img = gambar_hasil_deteksi(resized_rgb, pred_mask, hasil)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("CT Scan Asli")
        st.image(resized_rgb, use_container_width=True)
    with col2:
        st.subheader("Hasil Segmentasi & Lokalisasi")
        st.image(vis_img, use_container_width=True)

    st.divider()
    st.subheader("Hasil Analisis & Dimensi (Estimasi)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", "Terdeteksi" if hasil["bbox"] else "Normal")
    c2.metric("Lebar × Panjang", f"{hasil['lebar_mm']} × {hasil['panjang_mm']} mm")
    c3.metric("Approx. Volume", f"{hasil['approx_volume_ml']} mL")
    c4.metric("Letak Anatomi", hasil["letak_anatomi"])

    if hasil["subtipe"]:
        st.warning(f"**Indikasi Subtipe Pendarahan:** {', '.join(hasil['subtipe'])}")

    st.caption("Disclaimer: Nilai dimensi (mm) dan volume (mL) merupakan estimasi matematis dari citra 2D JPG (asumsi FOV 240 mm & ketebalan slice 5 mm), bukan pengganti pengukuran DICOM medis resmi.")