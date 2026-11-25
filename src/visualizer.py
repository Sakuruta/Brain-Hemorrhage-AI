import cv2
import numpy as np

# Warna RGB untuk masing-masing subtipe (1-5)
CLASS_COLORS = {
    1: (255, 0, 0),     # EDH: Merah
    2: (0, 100, 255),   # IPH: Biru
    3: (255, 0, 255),   # IVH: Magenta
    4: (255, 255, 0),   # SAH: Kuning
    5: (0, 255, 255)    # SDH: Cyan
}

def gambar_hasil_deteksi(original_rgb, pred_mask, hasil_metrik):
    img_vis = original_rgb.copy()
    overlay = img_vis.copy()

    for class_id, color in CLASS_COLORS.items():
        overlay[pred_mask == class_id] = color

    # Gabungkan warna overlay dengan gambar asli (transparansi 40%)
    cv2.addWeighted(overlay, 0.4, img_vis, 0.6, 0, img_vis)

    # Gambar kotak lokasi dan titik pusat pendarahan
    if hasil_metrik["bbox"] is not None:
        x, y, w, h = hasil_metrik["bbox"]
        cx, cy = hasil_metrik["koordinat_pusat"]
        cv2.rectangle(img_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(img_vis, (cx, cy), 4, (0, 255, 0), -1)

    return img_vis