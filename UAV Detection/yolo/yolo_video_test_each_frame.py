import cv2
import torch
import time
from ultralytics import YOLO

# Modeli yükle
model = YOLO("best.pt")

# Video dosyasını aç
video_path = "../data/test_chase_video.mp4"
cap = cv2.VideoCapture(video_path)

# Video bilgilerini al
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Çıkış videosu için ayarlar
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
output_path = "output_live.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# GPU kontrolü
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# FPS ölçümü için başlangıç zamanı
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 
    
    # Zaman ölçümü başlat
    frame_start = time.time()

    # Modeli çalıştır ve tahminleri al
    results = model(frame)[0]

    # Bounding box çiz
    for result in results.boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0]) 
        confidence = result.conf[0] 
        class_id = int(result.cls[0]) 
        label = f"{model.names[class_id]} {confidence:.2f}"

        # Bounding box çizimi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # FPS hesapla
    frame_end = time.time()
    frame_time = frame_end - frame_start
    fps_live = 1 / frame_time if frame_time > 0 else 0

    # FPS ekrana yazdır
    cv2.putText(frame, f"FPS: {fps_live:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Çıkış videosuna yaz
    out.write(frame)

    # Canlı önizleme
    cv2.imshow("YOLO Live Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

    frame_count += 1

# Toplam FPS hesapla
total_time = time.time() - start_time
avg_fps = frame_count / total_time if total_time > 0 else 0
print(f"Ortalama FPS: {avg_fps:.2f}")

# Temizlik
cap.release()
out.release()
cv2.destroyAllWindows()
