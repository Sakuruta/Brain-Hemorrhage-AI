import cv2
import numpy as np

CLASS_LABELS = {
    0: "Normal",
    1: "Epidural (EDH)",
    2: "Intraparenchymal (IPH)",
    3: "Intraventricular (IVH)",
    4: "Subarachnoid (SAH)",
    5: "Subdural (SDH)"
}

def ekstrak_analisis_pendarahan(pred_mask, img_size=256, fov_mm=240.0, slice_thickness_cm=0.5):
    """
    pred_mask: array 2D (256x256) berisi integer kelas (0 sampai 5)
    fov_mm: estimasi Field of View standar CT scan kepala (240 mm)
    slice_thickness_cm: estimasi ketebalan irisan CT standar (0.5 cm = 5 mm)
    """
    pixel_spacing_mm = fov_mm / img_size
    binary_mask = (pred_mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "status": "Normal (Tidak Terdeteksi Pendarahan)",
            "subtipe": [],
            "letak_anatomi": "-",
            "koordinat_pusat": None,
            "bbox": None,
            "lebar_mm": 0.0,
            "panjang_mm": 0.0,
            "approx_volume_ml": 0.0
        }

    # Identifikasi jenis subtipe pendarahan berdasarkan kelas piksel yang muncul
    unique_classes = np.unique(pred_mask)
    detected_subtypes = [CLASS_LABELS[c] for c in unique_classes if c in CLASS_LABELS and c != 0]

    # Ambil kontur pendarahan terbesar untuk perhitungan dimensi utama
    c = max(contours, key=cv2.contourArea)
    x, y, w_px, h_px = cv2.boundingRect(c)
    cx, cy = x + w_px // 2, y + h_px // 2

    # Estimasi lokasi anatomi (Perspektif radiologi: kiri gambar = sisi kanan pasien)
    sisi = "Hemisfer Kanan" if cx < (img_size // 2) else "Hemisfer Kiri"
    bagian = "Anterior (Depan)" if cy < (img_size // 2) else "Posterior (Belakang)"
    letak_anatomi = f"{sisi} - {bagian}"

    # Hitung dimensi fisik (mm)
    lebar_mm = w_px * pixel_spacing_mm
    panjang_mm = h_px * pixel_spacing_mm

    # Hitung Approximate Volume (mL) dengan pendekatan ABC/2
    A_cm = max(lebar_mm, panjang_mm) / 10.0
    B_cm = min(lebar_mm, panjang_mm) / 10.0
    C_cm = slice_thickness_cm
    approx_volume_ml = (A_cm * B_cm * C_cm) / 2.0

    return {
        "status": "Terdeteksi Pendarahan Otak",
        "subtipe": detected_subtypes,
        "letak_anatomi": letak_anatomi,
        "koordinat_pusat": (cx, cy),
        "bbox": (x, y, w_px, h_px),
        "lebar_mm": round(lebar_mm, 2),
        "panjang_mm": round(panjang_mm, 2),
        "approx_volume_ml": round(approx_volume_ml, 2)
    }