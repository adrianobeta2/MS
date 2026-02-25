
import cv2
import time
from ultralytics import YOLO
import threading

# ========= CONFIG =========
RTSP_URL = "rtsp://admin:V@ssoura1331@192.168.1.64:554/Streaming/Channels/101"

MODEL_PATH = "yolov8s.pt"
IMG_SIZE = 416
CONF = 0.4

latest_frame = None
lock = threading.Lock()
running = True

def rtsp_reader():
    global latest_frame, running

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        with lock:
            latest_frame = frame

    cap.release()

print("🧠 Carregando YOLO (CPU)...")
model = YOLO(MODEL_PATH)

threading.Thread(target=rtsp_reader, daemon=True).start()

last_time = time.time()
fps = 0

while True:
    with lock:
        if latest_frame is None:
            continue
        frame = latest_frame.copy()

    t0 = time.time()

    results = model(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF,
        classes=[0],      # só pessoa
        device="cpu",
        verbose=False
    )

    infer_ms = (time.time() - t0) * 1000

    annotated = results[0].plot()

    now = time.time()
    fps = 1 / (now - last_time)
    last_time = now

    cv2.putText(
        annotated,
        f"FPS: {fps:.2f} | Infer: {infer_ms:.0f} ms",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLO CPU - Low Latency", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cv2.destroyAllWindows()