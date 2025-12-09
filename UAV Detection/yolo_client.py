# yolo_client.py
import socket
print("Imported socket successfully.")

import struct
print("Imported struct successfully.")

import numpy as np
print("Imported numpy successfully.")

import cv2
print("Imported OpenCV (cv2) successfully.")

import torch
print("Imported torch successfully.")

from ultralytics import YOLO
print("Imported ultralytics YOLO successfully.")

HOST, PORT = '127.0.0.1', 5599
DEVICE = 'cuda'

# --- DEĞİŞİKLİK BURADA YAPILDI ---
CONF = 0.5  # Confidence threshold 0.25'ten 0.5'e çekildi
# ---------------------------------

model = YOLO('yolo11n.pt')
print("DEBUG DEBUG DEBUG")

def recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

def parse_dims(hdr):
    # Try native (=) first, then network/big-endian (!) if needed
    for fmt in ('=HH', '!HH'):
        w, h = struct.unpack(fmt, hdr)
        if 16 <= w <= 4096 and 16 <= h <= 4096:
            return w, h
    return None, None

with socket.create_connection((HOST, PORT)) as s:
    cv2.namedWindow('YOLO (stream)', cv2.WINDOW_NORMAL)
    while True:
        hdr = recv_exact(s, 4)
        if hdr is None: break
        w, h = parse_dims(hdr)
        if w is None:
            print('Bad header, could not parse width/height'); break

        # Stream is grayscale: w*h bytes
        img_bytes = recv_exact(s, w * h)
        if img_bytes is None: break

        gray = np.frombuffer(img_bytes, dtype=np.uint8).reshape(h, w)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Tahmin (predict) sırasında CONF değişkeni kullanılıyor
        res = model.predict(source=rgb, device=DEVICE, conf=CONF, verbose=False)[0]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        names = model.names

        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Çizim işlemleri
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(bgr, f'{names.get(cls, cls)} {conf:.2f}',
                            (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('YOLO (stream)', bgr)
        if cv2.waitKey(1) == 27:  # ESC to quit
            break
