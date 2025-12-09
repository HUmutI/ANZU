import socket
import struct
import numpy as np
import cv2
import time
from ultralytics import YOLO
from pymavlink import mavutil

# --- AYARLAR ---
HOST, PORT = '127.0.0.1', 5599
DEVICE = 'cuda'
CONF = 0.15
MAV_CONNECTION_STRING = 'udp:127.0.0.1:14550'

# Kontrol Parametreleri
KP = 0.002           # Hız katsayısı (Çok hızlıysa düşür, çok yavaşsa artır)
MAX_SPEED = 1.0      # Drone'un yatayda yapacağı maksimum hız (m/s)
CENTER_TOLERANCE = 40 # Pixel cinsinden kabul edilebilir hata payı (Merkeze ne kadar yaklaşsın?)

# Durum Tanımları
STATE_SEARCHING = 0   # İnsan aranıyor
STATE_CENTERING = 1   # İnsan ortalanıyor
STATE_DESCENDING = 2  # Alçalıyor
STATE_ASCENDING = 3   # Yükseliyor
current_state = STATE_SEARCHING

# Zamanlayıcılar
maneuver_start_time = 0
DESCEND_DURATION = 4.0 # Saniye cinsinden alçalma süresi
ASCEND_DURATION = 4.0  # Saniye cinsinden yükselme süresi

model = YOLO('yolo11n.pt')

# --- MAVLINK FONKSİYONLARI ---
print(f"Drone aranıyor ({MAV_CONNECTION_STRING})...")
try:
    master = mavutil.mavlink_connection(MAV_CONNECTION_STRING)
    master.wait_heartbeat(timeout=5)
    print("Drone ile bağlantı kuruldu!")
    # GUIDED moduna al (Velocity komutları için gereklidir)
    mode_id = master.mode_mapping().get('GUIDED')
    master.set_mode(mode_id)
except Exception as e:
    print(f"HATA: Drone bağlanamadı! {e}")
    master = None

def send_body_velocity(vx, vy, vz):
    """
    Drone'un KENDİ gövdesine göre hız gönderir (Body Frame).
    vx: İleri (+) / Geri (-) [m/s]
    vy: Sağ (+) / Sol (-) [m/s]
    vz: Aşağı (+) / Yukarı (-) [m/s] (NED formatı olduğu için aşağı pozitiftir)
    """
    if master is None: return
    
    master.mav.set_position_target_local_ned_send(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_NED, # Gövdeye göre hareket et
        0b0000111111000111, # Sadece Hızları (Velocity) kullan maskesi
        0, 0, 0,       # Pos x, y, z (Kullanılmıyor)
        vx, vy, vz,    # Vel x, y, z
        0, 0, 0,       # Acc x, y, z (Kullanılmıyor)
        0, 0           # Yaw, Yaw rate (Kullanılmıyor)
    )

def recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk: return None
        buf += chunk
    return buf

def parse_dims(hdr):
    for fmt in ('=HH', '!HH'):
        w, h = struct.unpack(fmt, hdr)
        if 16 <= w <= 4096 and 16 <= h <= 4096: return w, h
    return None, None

# --- ANA DÖNGÜ ---
with socket.create_connection((HOST, PORT)) as s:
    cv2.namedWindow('YOLO (stream)', cv2.WINDOW_NORMAL)
    
    while True:
        hdr = recv_exact(s, 4)
        if hdr is None: break
        w, h = parse_dims(hdr)
        if w is None: break
        img_bytes = recv_exact(s, w * h)
        if img_bytes is None: break

        gray = np.frombuffer(img_bytes, dtype=np.uint8).reshape(h, w)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Görüntü merkezi
        center_x, center_y = w // 2, h // 2

        # Tahmin
        res = model.predict(source=rgb, device=DEVICE, conf=CONF, verbose=False, max_det=10)[0]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        target_box = None
        
        # İnsan tespiti
        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                cls = int(box.cls[0])
                if model.names[cls] == 'person':
                    target_box = box.xyxy[0].cpu().numpy().astype(int)
                    break # İlk bulduğun insana odaklan

        # --- DURUM MAKİNESİ (STATE MACHINE) ---
        
        if target_box is not None:
            x1, y1, x2, y2 = target_box
            # Kutunun merkezini bul
            box_cx = (x1 + x2) // 2
            box_cy = (y1 + y2) // 2
            
            # Görselleştirme
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(bgr, (box_cx, box_cy), 5, (0, 255, 255), -1)
            cv2.line(bgr, (center_x, center_y), (box_cx, box_cy), (255, 0, 0), 2)

            # Hata hesaplama (Hedef - Merkez)
            error_x = box_cx - center_x
            error_y = box_cy - center_y  # Not: Görüntüde Y aşağı doğru artar

            # === STATE: CENTERING ===
            if current_state == STATE_SEARCHING or current_state == STATE_CENTERING:
                current_state = STATE_CENTERING
                
                # Kamera aşağı baktığı için:
                # Görüntüde SAĞA gitmek (error_x > 0) -> Drone SAĞA (Vy > 0)
                # Görüntüde AŞAĞI gitmek (error_y > 0) -> Drone GERİ (Vx < 0) !!!
                # (Kameranın üst kısmını dronun burnu kabul ediyoruz)
                
                # PID benzeri oransal kontrol
                vel_y = -error_x * KP
                vel_x = error_y * KP  # İşaret eksi çünkü görüntüde aşağısı dronun arkasıdır.

                # Hız Limitleme
                vel_x = max(min(vel_x, MAX_SPEED), -MAX_SPEED)
                vel_y = max(min(vel_y, MAX_SPEED), -MAX_SPEED)

                # Eğer hedefe çok yakınsak dur ve sonraki aşamaya geç
                if abs(error_x) < CENTER_TOLERANCE and abs(error_y) < CENTER_TOLERANCE:
                    print(">>> HEDEF ORTALANDI! İniş başlıyor...")
                    send_body_velocity(0, 0, 0) # Dur
                    current_state = STATE_DESCENDING
                    maneuver_start_time = time.time()
                else:
                    cv2.putText(bgr, f"Hizalaniyor... Vx:{vel_x:.2f} Vy:{vel_y:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    send_body_velocity(vel_x, vel_y, 0)

            # === STATE: DESCENDING ===
            elif current_state == STATE_DESCENDING:
                if time.time() - maneuver_start_time < DESCEND_DURATION:
                    cv2.putText(bgr, "ALCALIYOR...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # Vz pozitif = AŞAĞI
                    send_body_velocity(0, 0, 0.5) # 0.5 m/s ile alçal
                else:
                    print(">>> Alçalma bitti, yükseliyor...")
                    current_state = STATE_ASCENDING
                    maneuver_start_time = time.time()

            # === STATE: ASCENDING ===
            elif current_state == STATE_ASCENDING:
                if time.time() - maneuver_start_time < ASCEND_DURATION:
                    cv2.putText(bgr, "YUKSELIYOR...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # Vz negatif = YUKARI
                    send_body_velocity(0, 0, -0.5) # 0.5 m/s ile yüksel
                else:
                    print(">>> Görev Tamamlandı! Başa dönülüyor.")
                    send_body_velocity(0, 0, 0)
                    current_state = STATE_SEARCHING # Veya bitirmek istersen break
                    
        else:
            # İnsan görülmediğinde dur ya da aramaya devam et
            if current_state == STATE_CENTERING:
                send_body_velocity(0, 0, 0) # Kaybettik, dur.
                print("Hedef kayboldu, duruluyor.")
                current_state = STATE_SEARCHING

        # Merkez noktayı çiz (Referans için)
        cv2.circle(bgr, (center_x, center_y), 5, (0, 0, 255), -1)
        
        cv2.imshow('YOLO (stream)', bgr)
        if cv2.waitKey(1) == 27: break
