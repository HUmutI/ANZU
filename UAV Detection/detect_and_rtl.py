# yolo_client.py
import socket
import struct
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# --- MAVLINK EKLENDİ ---
from pymavlink import mavutil
import time
# -----------------------

HOST, PORT = '127.0.0.1', 5599
DEVICE = 'cuda'
CONF = 0.5

# --- DRONE BAĞLANTI AYARLARI ---
# ArduPilot SITL genellikle 14550 veya 14551 portundan UDP yayını yapar.
# Webots simülasyonunun dışarıya verdiği portu buraya yazmalısın.
MAV_CONNECTION_STRING = 'udp:127.0.0.1:14550' 
# -------------------------------

print("YOLO modeli yükleniyor...")
model = YOLO('yolo11n.pt')

# --- DRONE'A BAĞLANMA ---
print(f"Drone aranıyor ({MAV_CONNECTION_STRING})...")
try:
    master = mavutil.mavlink_connection(MAV_CONNECTION_STRING)
    # İlk kalp atışını bekle (bağlantıyı doğrula)
    master.wait_heartbeat(timeout=5)
    print(f"Drone bulundu! Sistem ID: {master.target_system}, Bileşen ID: {master.target_component}")
except Exception as e:
    print(f"UYARI: Drone'a bağlanılamadı. Sadece görüntü işlenecek. Hata: {e}")
    master = None

rtl_triggered = False  # RTL komutunun sadece bir kez gönderilmesi için bayrak

def set_mode_rtl(mav_conn):
    """Drone'u RTL moduna geçirir"""
    if mav_conn is None:
        return
    
    # ArduPilot için mod değiştirme güvenli yolu
    mode_id = mav_conn.mode_mapping().get('RTL')
    if mode_id is None:
        print("HATA: RTL modu bu araç tipinde bulunamadı!")
        return

    print(">>> KOMUT GÖNDERİLİYOR: RTL MODU <<<")
    mav_conn.set_mode(mode_id)

def recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk: return None
        buf += chunk
    return buf


#big endian mı little endian mı?
def parse_dims(hdr):
    for fmt in ('=HH', '!HH'):
        w, h = struct.unpack(fmt, hdr)
        if 16 <= w <= 4096 and 16 <= h <= 4096:
            return w, h
    return None, None

with socket.create_connection((HOST, PORT)) as s:
    cv2.namedWindow('YOLO (stream)', cv2.WINDOW_NORMAL)
    print("Video akışı başladı.")
    
    while True:
        hdr = recv_exact(s, 4)
        if hdr is None: break
        w, h = parse_dims(hdr)
        if w is None:
            print('Bad header'); break

        img_bytes = recv_exact(s, w * h)
        if img_bytes is None: break

        gray = np.frombuffer(img_bytes, dtype=np.uint8).reshape(h, w)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        res = model.predict(source=rgb, device=DEVICE, conf=CONF, verbose=False)[0]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        names = model.names

        person_detected_in_frame = False

        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = names.get(cls, cls)

                # --- MANTIK BURADA ---
                # Eğer tespit edilen sınıf 'person' ise (COCO veri setinde person genelde ID 0'dır ama isme bakmak daha garanti)
                if label == 'person':
                    person_detected_in_frame = True
                    # Ekrana kırmızı kutu çizelim dikkat çekmek için
                    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(bgr, f'INSAN TESPIT EDILDI! RTL', (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(bgr, f'{label} {conf:.2f}', (x1, y1-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # RTL Tetikleme Mantığı
        if person_detected_in_frame and not rtl_triggered:
            if master:
                set_mode_rtl(master)
                rtl_triggered = True  # Tekrar tekrar komut göndermeyi engelle
                print("RTL komutu gönderildi.")
        
        # İsteğe bağlı: İnsan kadrajdan çıkarsa bayrağı sıfırlamak istersen:
        # if not person_detected_in_frame:
        #     rtl_triggered = False 

        cv2.imshow('YOLO (stream)', bgr)
        if cv2.waitKey(1) == 27:
            break
