import os
import shutil
import random

# Ana klasör ve alt klasörler
data_dir = "../data/anzu_data"
image_dir = os.path.join(data_dir, "images")
label_dir = os.path.join(data_dir, "labels")

# Çıktı klasörleri
output_dir = "../data/anzu_data"
train_image_dir = os.path.join(output_dir, "train", "images")
train_label_dir = os.path.join(output_dir, "train", "labels")
val_image_dir = os.path.join(output_dir, "val", "images")
val_label_dir = os.path.join(output_dir, "val", "labels")

# Çıktı klasörlerini oluştur
for folder in [train_image_dir, train_label_dir, val_image_dir, val_label_dir]:
    os.makedirs(folder, exist_ok=True)

# Tüm resim dosyalarını al (YOLO formatında olduğundan eşleşen .txt etiketleri olmalı)
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(image_files)  # Karıştır

# %90 train, %10 val ayır
split_index = int(len(image_files) * 0.9)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

def copy_files(file_list, dest_image_dir, dest_label_dir):
    for file in file_list:
        image_path = os.path.join(image_dir, file)
        label_path = os.path.join(label_dir, file.replace(os.path.splitext(file)[1], ".txt"))
        
        # Resim ve etiket dosyasını kopyala
        shutil.copy(image_path, os.path.join(dest_image_dir, file))
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(dest_label_dir, os.path.basename(label_path)))

# Dosyaları ilgili klasörlere kopyala
copy_files(train_files, train_image_dir, train_label_dir)
copy_files(val_files, val_image_dir, val_label_dir)

print("Done.")