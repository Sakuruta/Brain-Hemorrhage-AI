import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class BrainHemorrhageDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=256):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        # Mengambil hanya file gambar agar tidak error jika ada file sistem tersembunyi
        valid_ext = ('.jpg', '.jpeg', '.png')
        self.filenames = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(valid_ext)])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_name)

        # 1. Baca gambar CT Scan (.jpg)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype(np.float32) / 255.0

        # Normalisasi standar ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))  # Ubah urutan ke (Channel, Height, Width)

        # 2. Baca mask diagnosis (.png)
        # INTER_NEAREST wajib dipakai agar angka kelas piksel (0-5) tidak tercampur
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)