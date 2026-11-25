import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from src.dataset import BrainHemorrhageDataset

# Konfigurasi Training
BATCH_SIZE = 16
IMAGE_SIZE = 256
NUM_EPOCHS = 20
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Menggunakan perangkat komputasi: {DEVICE}")

    train_dataset = BrainHemorrhageDataset('./dataset/train/images', './dataset/train/masks', IMAGE_SIZE)
    val_dataset = BrainHemorrhageDataset('./dataset/val/images', './dataset/val/masks', IMAGE_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Inisialisasi Lightweight U-Net dengan backbone MobileNetV2
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES
    ).to(DEVICE)

    # Bobot moderat hasil kalibrasi dari distribusi hemorrhage_diagnosis.csv:
    # [0: Normal, 1: EDH, 2: IPH, 3: IVH, 4: SAH, 5: SDH]
    class_weights = torch.tensor([1.0, 3.5, 5.5, 10.0, 12.0, 6.5], dtype=torch.float32).to(DEVICE)

    # Kombinasi Weighted CrossEntropy & DiceLoss agar peka pada bercak pendarahan kecil
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    os.makedirs('./models', exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = ce_loss(outputs, masks) + dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)

        # Fase Validasi
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = ce_loss(outputs, masks) + dice_loss(outputs, masks)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_dataset)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), './models/best_unet.pt')
            print("  --> Model terbaik disimpan ke ./models/best_unet.pt")

if __name__ == '__main__':
    main()