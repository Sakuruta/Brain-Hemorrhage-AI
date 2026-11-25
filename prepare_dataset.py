import os
import csv
import random
import shutil
import cv2
import numpy as np

RAW_DATA_DIR = './Patients_CT'
CSV_PATH = './hemorrhage_diagnosis.csv'
OUTPUT_DIR = './dataset'
TRAIN_RATIO = 0.8
SEED = 42

SUBTYPE_TO_CLASS = [
    ('Epidural', 1),
    ('Intraparenchymal', 2),
    ('Intraventricular', 3),
    ('Subarachnoid', 4),
    ('Subdural', 5),
]

def load_diagnosis_map(csv_path):
    diag_map = {}
    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row['PatientNumber'])
            sid = int(row['SliceNumber'])
            class_id = 0
            for col_name, cid in SUBTYPE_TO_CLASS:
                if int(row.get(col_name, 0)) == 1:
                    class_id = cid
                    break
            diag_map[(pid, sid)] = class_id
    return diag_map

def prepare():
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Folder '{RAW_DATA_DIR}' tidak ditemukan.")
        return
    if not os.path.exists(CSV_PATH):
        print(f"Error: File '{CSV_PATH}' tidak ditemukan. Pastikan sudah disalin ke folder proyek.")
        return

    diag_map = load_diagnosis_map(CSV_PATH)

    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'masks'), exist_ok=True)

    patient_ids = sorted([
        d for d in os.listdir(RAW_DATA_DIR)
        if os.path.isdir(os.path.join(RAW_DATA_DIR, d))
    ])

    random.seed(SEED)
    random.shuffle(patient_ids)

    split_idx = int(len(patient_ids) * TRAIN_RATIO)
    splits = {
        'train': patient_ids[:split_idx],
        'val': patient_ids[split_idx:]
    }

    total_copied = {'train': 0, 'val': 0}

    for split_name, p_list in splits.items():
        for pid_str in p_list:
            brain_dir = os.path.join(RAW_DATA_DIR, pid_str, 'brain')
            if not os.path.exists(brain_dir):
                continue

            pid_int = int(pid_str)
            all_files = os.listdir(brain_dir)
            ct_slices = [f for f in all_files if f.lower().endswith('.jpg') and '_HGE_Seg' not in f]

            for slice_file in ct_slices:
                base_name = os.path.splitext(slice_file)[0]
                sid_int = int(base_name)
                new_filename = f"{pid_str}_{base_name}"

                src_img_path = os.path.join(brain_dir, slice_file)
                dst_img_path = os.path.join(OUTPUT_DIR, split_name, 'images', f"{new_filename}.jpg")
                dst_mask_path = os.path.join(OUTPUT_DIR, split_name, 'masks', f"{new_filename}.png")

                shutil.copy2(src_img_path, dst_img_path)

                src_mask_path = os.path.join(brain_dir, f"{base_name}_HGE_Seg.jpg")
                if os.path.exists(src_mask_path):
                    mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
                    class_id = diag_map.get((pid_int, sid_int), 2)
                    if class_id == 0:
                        class_id = 2
                    multi_mask = np.where(mask > 127, class_id, 0).astype(np.uint8)
                else:
                    img_temp = cv2.imread(src_img_path, cv2.IMREAD_GRAYSCALE)
                    multi_mask = np.zeros_like(img_temp, dtype=np.uint8)

                cv2.imwrite(dst_mask_path, multi_mask)
                total_copied[split_name] += 1

    print("Proses persiapan dataset 6 kelas selesai!")
    print(f"Total Train : {len(splits['train'])} pasien ({total_copied['train']} slices)")
    print(f"Total Val   : {len(splits['val'])} pasien ({total_copied['val']} slices)")

if __name__ == '__main__':
    prepare()