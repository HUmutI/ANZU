from ultralytics import YOLO

models = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]

data_yaml = "dataset.yaml"

epochs = 120
batch_size = 32
img_size = 640
device = "cuda"

for model_name in models:
    print(f"\n{model_name} modeli eğitiliyor...")
    
    model = YOLO(model_name)    
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project="yolo11_models", 
        name=model_name.split(".")[0],
    )
    
    print(f"{model_name} modeli eğitimi tamamlandı.\n")

print("Tüm modellerin eğitimi tamamlandı.")
