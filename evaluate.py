import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from src.dataset import BrainHemorrhageDataset

IMAGE_SIZE = 256
NUM_CLASSES = 6
BATCH_SIZE = 16
MODEL_PATH = './models/best_unet.pt'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Normal (0)",
    "Epidural (1)",
    "IPH (2)",
    "IVH (3)",
    "SAH (4)",
    "Subdural (5)"
]

def evaluate():
    print(f"Mengevaluasi model {MODEL_PATH} pada {DEVICE}...")
    val_dataset = BrainHemorrhageDataset('./dataset/val/images', './dataset/val/masks', IMAGE_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = smp.Unet(encoder_name="mobilenet_v2", in_channels=3, classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = masks.numpy()

            for p, t in zip(preds.flatten(), targets.flatten()):
                conf_matrix[t, p] += 1

    os.makedirs('./evaluation_results', exist_ok=True)

    print("\n=== LAPORAN METRIK SEGMENTASI PER KELAS (VALIDATION SET) ===")
    print(f"{'Kelas':<16} | {'Dice/F1':<9} | {'IoU':<9} | {'Recall (Sens)':<13} | {'Precision':<9}")
    print("-" * 68)

    dice_scores = []
    for c in range(NUM_CLASSES):
        tp = conf_matrix[c, c]
        fp = conf_matrix[:, c].sum() - tp
        fn = conf_matrix[c, :].sum() - tp

        dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
        iou = tp / (tp + fp + fn + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        precision = tp / (tp + fp + 1e-7)

        if c > 0:  # Rata-rata khusus kelas pendarahan (1-5)
            dice_scores.append(dice)

        print(f"{CLASS_NAMES[c]:<16} | {dice*100:7.2f}% | {iou*100:7.2f}% | {recall*100:11.2f}% | {precision*100:7.2f}%")

    print("-" * 68)
    print(f"Mean Dice Score (Khusus 5 Subtipe Pendarahan): {np.mean(dice_scores)*100:.2f}%")

    # Simpan Plot Confusion Matrix Ternormalisasi per Baris (%)
    row_sums = conf_matrix.sum(axis=1, keepdims=True) + 1e-7
    norm_cm = (conf_matrix / row_sums) * 100

    plt.figure(figsize=(9, 7))
    sns.heatmap(norm_cm, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Normalized Pixel-Level Confusion Matrix (%) - Validation Set")
    plt.ylabel("Ground Truth (Dokter)")
    plt.xlabel("Prediksi Model (U-Net + MobileNetV2)")
    plt.tight_layout()
    plt.savefig('./evaluation_results/confusion_matrix.png', dpi=300)
    print("\nGrafik Confusion Matrix disimpan ke: ./evaluation_results/confusion_matrix.png")

if __name__ == '__main__':
    evaluate()