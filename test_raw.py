import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from src.metrics import ekstrak_analisis_pendarahan
from src.visualizer import gambar_hasil_deteksi

IMAGE_PATH = './test_images/sample_ct.jpg'
MODEL_PATH = './models/best_unet.pt'
IMAGE_SIZE = 256
NUM_CLASSES = 6  # Pastikan sama dengan NUM_CLASSES di train.py
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_single_image(img_path):
    model = smp.Unet(encoder_name="mobilenet_v2", in_channels=3, classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    # Preprocessing gambar uji
    raw_bgr = cv2.imread(img_path)
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(raw_rgb, (IMAGE_SIZE, IMAGE_SIZE))

    inp = resized_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = np.transpose((inp - mean) / std, (2, 0, 1))
    tensor_inp = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Inferensi
    with torch.no_grad():
        logits = model(tensor_inp)
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    # Hitung metrik dan buat visualisasi
    hasil = ekstrak_analisis_pendarahan(pred_mask, img_size=IMAGE_SIZE)
    vis_img = gambar_hasil_deteksi(resized_rgb, pred_mask, hasil)

    print("\n=== HASIL ANALISIS CT SCAN ===")
    for k, v in hasil.items():
        if k != "bbox":
            print(f"{k}: {v}")

    output_path = './test_images/hasil_prediksi.jpg'
    cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    print(f"\nGambar visualisasi tersimpan di: {output_path}")

if __name__ == '__main__':
    predict_single_image(IMAGE_PATH)