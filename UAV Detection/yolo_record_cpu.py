# yolo_csi_detect_record.py
from ultralytics import YOLO
import cv2
from datetime import datetime

# ---------------------------
# 1️⃣ GStreamer pipeline (capture)
# ---------------------------
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    framerate=30,
    flip_method=0,
    sensor_id=0,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"framerate=(fraction){framerate}/1, format=(string)NV12 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        # Critical for stability on Jetson
        "appsink drop=true max-buffers=1 sync=false"
    )

# ---------------------------
# 1️⃣b GStreamer pipeline (hardware writer)
# ---------------------------
def gstreamer_writer_pipeline(width, height, framerate, filepath, bitrate=8000000):
    return (
        "appsrc ! "
        "videoconvert ! "
        f"video/x-raw,format=BGR,width={width},height={height},framerate={framerate}/1 ! "
        "nvvidconv ! "
        "video/x-raw(memory:NVMM),format=NV12 ! "
        f"nvv4l2h264enc insert-sps-pps=true bitrate={bitrate} maxperf-enable=1 ! "
        "h264parse ! "
        "qtmux ! "
        f"filesink location={filepath} sync=false"
    )

# ---------------------------
# 2️⃣ Class filtering
# ---------------------------
ALLOWED_CLASSES = {
    "person", "car", "bicycle", "motorcycle", "bus", "truck",
    "chair", "couch", "bed", "cat", "dog"
}
EXCLUDED_CLASSES = set()

# ---------------------------
# 3️⃣ Optional per-class handlers
# ---------------------------
def handle_person(): pass
def handle_car(): pass
def handle_dog(): pass
def handle_cat(): pass
def handle_chair(): pass

CLASS_FUNCTIONS = {
    "person": handle_person,
    "car": handle_car,
    "dog": handle_dog,
    "cat": handle_cat,
    "chair": handle_chair,
}

def class_allowed(name: str) -> bool:
    if ALLOWED_CLASSES and name not in ALLOWED_CLASSES:
        return False
    if name in EXCLUDED_CLASSES:
        return False
    return True

def draw_detections(frame, detections):
    for (x1, y1, x2, y2, class_name, score) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} ({score:.2f})"
        cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        func = CLASS_FUNCTIONS.get(class_name)
        if func:
            func()

    return frame

def detect_and_record():
    print("🚀 Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    print("✅ Model loaded successfully")

    # --- Capture settings (safe defaults for Orin Nano) ---
    CAP_W, CAP_H = 1280, 720
    FPS = 30
    SENSOR_ID = 0
    FLIP = 0

    cap = cv2.VideoCapture(
        gstreamer_pipeline(
            capture_width=CAP_W,
            capture_height=CAP_H,
            framerate=FPS,
            flip_method=FLIP,
            sensor_id=SENSOR_ID
        ),
        cv2.CAP_GSTREAMER
    )

    if not cap.isOpened():
        print("❌ Failed to open CSI camera!")
        return

    # --- Prepare output file name ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = f"raw_{ts}.mp4"

    writer = None
    print("🎥 CSI camera stream started...")
    print(f"⏺️  Raw recording will be saved to: {raw_path}")
    print("Press 'q' to quit.")

    # --- Display settings ---
    DISPLAY_W, DISPLAY_H = 960, 540

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame read failed. Check camera connection.")
            break

        # Initialize HW writer once we know actual frame size
        if writer is None:
            h, w = frame.shape[:2]
            out_pipe = gstreamer_writer_pipeline(w, h, FPS, raw_path)
            writer = cv2.VideoWriter(out_pipe, cv2.CAP_GSTREAMER, 0, FPS, (w, h), True)

            # Fallback to CPU mp4v if HW writer fails
            if not writer.isOpened():
                print("⚠️ HW VideoWriter failed, falling back to CPU mp4v...")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(raw_path, fourcc, FPS, (w, h))

                if not writer.isOpened():
                    print("❌ VideoWriter failed to open (HW + CPU).")
                    break

        # ✅ 1) Save RAW (unaltered)
        writer.write(frame)

        # ✅ 2) YOLO inference
        results = model(frame, verbose=False, conf=0.5)

        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names.get(cls_id, "unknown")
                if not class_allowed(class_name):
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((x1, y1, x2, y2, class_name, conf))

        # Draw on a separate view frame (so raw stays clean)
        view = frame.copy()
        view = draw_detections(view, detections)

        # Optional resize for display
        view_small = cv2.resize(view, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("YOLO CSI Camera Detection", view_small)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("🟢 Detection + raw recording ended cleanly.")

if __name__ == "__main__":
    detect_and_record()
