import cv2
import time
import os
import torch
from ultralytics import YOLO
from depth_estimator import DepthEstimator

def load_yolo_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path).to(device)
    return model

def process_video(video_path, model_name, model, precision, focal_length, output_folder="output_videos"):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{model_name}_with_depth.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_times = []
    device = next(model.model.parameters()).device
    dtype = torch.float16 if precision == "fp16" else torch.float32

    # DepthEstimator oluştur
    depth_estimator = DepthEstimator(focal_length=focal_length)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        resized_frame = cv2.resize(frame, (640, 640))
        img = torch.from_numpy(resized_frame).to(device).permute(2, 0, 1).unsqueeze(0).to(dtype) / 255.0
        
        start_time = time.time()
        results = model.predict(img, imgsz=640)
        end_time = time.time()
        
        frame_times.append(end_time - start_time)
        if len(frame_times) > fps:
            frame_times.pop(0)

        # Güncellenmiş draw_detections fonksiyonunu çağır
        resized_frame = draw_detections(resized_frame, results, depth_estimator)
        frame = cv2.resize(resized_frame, (width, height))
        
        display_fps(frame, frame_times, model_name)
        out.write(frame)
        
    cap.release()
    out.release()

def draw_detections(frame, results, depth_estimator):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()  # Tensor'i float yapıyoruz
            cls = int(box.cls[0].item()) if hasattr(box, 'cls') else 0
            
            # Derinliği hesapla
            depth = depth_estimator.estimate_depth([x1, y1, x2, y2])

            # Kutu çizimi
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Metin (conf ve depth birlikte)
            label = f"Conf: {conf:.2f} | Depth: {depth:.2f}m"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame

def display_fps(frame, frame_times, model_name):
    avg_time = sum(frame_times) / len(frame_times) if frame_times else 0
    fps = 1 / avg_time if avg_time > 0 else 0
    text = f"{model_name} | FPS: {fps:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
