from ultralytics import YOLO
from tqdm import tqdm

model = YOLO("yolo11n.pt") 

data_yaml = "dataset.yaml"  

epochs = 50
model.train(
    data=data_yaml,      # Dataset yaml dosyası (train/val klasör yollarını içermeli)
    epochs=epochs,       # Eğitim epoch sayısı
    batch=16,            # Batch size
    imgsz=640,           # Görüntü boyutu
    device="cuda",       # GPU kullanımı
    )

print("Done.")
