import os
import random
import shutil

# Klasör yolları
dataset_dir = "datasets/"
train_img_dir = os.path.join(dataset_dir, "train/images")
train_label_dir = os.path.join(dataset_dir, "train/labels")

valid_img_dir = os.path.join(dataset_dir, "valid/images")
valid_label_dir = os.path.join(dataset_dir, "valid/labels")

test_img_dir = os.path.join(dataset_dir, "test/images")
test_label_dir = os.path.join(dataset_dir, "test/labels")

# Train setindeki tüm görüntü dosyalarını al
image_files = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png'))]

# Eğer 2000'den az görüntü varsa, hata verme ama uyar
if len(image_files) < 2000:
    print(f"⚠️ Uyarı: Train setinde {len(image_files)} görüntü var, 2000 örnek ayıramayabilirim!")

# Train setinden rastgele 2000 görüntü seç
random.shuffle(image_files)
valid_samples = image_files[:1000]  # İlk 1000 valid
test_samples = image_files[1000:2000]  # Sonraki 1000 test

# Fonksiyon: Dosyaları taşı
def move_files(samples, src_img_dir, src_label_dir, dest_img_dir, dest_label_dir):
    for img_file in samples:
        base_name = os.path.splitext(img_file)[0]  # Dosya adı (uzantısız)
        img_path = os.path.join(src_img_dir, img_file)
        label_path = os.path.join(src_label_dir, base_name + ".txt")

        # Eğer etiket dosyası yoksa, bu görüntüyü atla
        if not os.path.exists(label_path):
            print(f"⚠️ {label_path} etiketi bulunamadı, atlanıyor...")
            continue

        # Yeni hedef yollar
        new_img_path = os.path.join(dest_img_dir, img_file)
        new_label_path = os.path.join(dest_label_dir, base_name + ".txt")

        # Dosyaları taşı
        shutil.move(img_path, new_img_path)
        shutil.move(label_path, new_label_path)

# Valid setine taşı
move_files(valid_samples, train_img_dir, train_label_dir, valid_img_dir, valid_label_dir)

# Test setine taşı
move_files(test_samples, train_img_dir, train_label_dir, test_img_dir, test_label_dir)

print("✅ Train setinden 1000'er örnek valid ve test setlerine taşındı.")
