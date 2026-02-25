import cv2
import threading
import time

from core.frame_store import FrameStore
from cameras.camera_stream import CameraStream
from monitorar_movimentos import monitor_movimento, monitor_movimento_carrinho

# =========================
# CONFIG
# =========================
FPS_LIMIT = 15

WINDOW_1 = "Camera 1 - Berço"
WINDOW_2 = "Camera 2 - Carrinho"

running = True
frame_berco = None
frame_carrinho = None

lock_berco = threading.Lock()
lock_carrinho = threading.Lock()

# =========================
# FRAME STORE
# =========================
frame_store = FrameStore()

CAMERAS = {
    "cam1":  "rtsp://admin:V@ssoura1331@192.168.1.64:554/Streaming/Channels/101"
    #"cam2":  "rtsp://admin:V@ssoura1331@192.168.1.64:554/Streaming/Channels/101"
}

streams = []

for cam_id, src in CAMERAS.items():
    stream = CameraStream(
        cam_id=cam_id,
        src=src,
        frame_store=frame_store,
        width=640,
        height=480
    )
    stream.start()
    streams.append(stream)

print("🎥 Câmeras iniciadas, aguardando frames...")

# =========================
# THREAD CAMERA 1
# =========================
def thread_camera_berco():
    global frame_berco, running

    interval = 1.0 / FPS_LIMIT
    last_time = 0

    while running:
        now = time.time()
        if now - last_time < interval:
            time.sleep(0.001)
            continue
        last_time = now

        frame = frame_store.get_frame("cam1")
        if frame is None:
            continue

        annotated, *_ = monitor_movimento(frame, 1, 1)

        with lock_berco:
            frame_berco = annotated.copy()

# =========================
# THREAD CAMERA 2
# =========================
def thread_camera_carrinho():
    global frame_carrinho, running

    interval = 1.0 / FPS_LIMIT
    last_time = 0

    while running:
        now = time.time()
        if now - last_time < interval:
            time.sleep(0.001)
            continue
        last_time = now

        frame = frame_store.get_frame("cam2")
        if frame is None:
            continue

        annotated, _ = monitor_movimento_carrinho(frame, 1, 1)

        with lock_carrinho:
            frame_carrinho = annotated.copy()

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    cv2.namedWindow(WINDOW_1, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_2, cv2.WINDOW_NORMAL)

    t1 = threading.Thread(target=thread_camera_berco, daemon=True)
    t2 = threading.Thread(target=thread_camera_carrinho, daemon=True)

    t1.start()
    t2.start()

    print("[INFO] Pressione Q para sair")

    while running:
        if frame_berco is not None:
            with lock_berco:
                cv2.imshow(WINDOW_1, frame_berco)

        if frame_carrinho is not None:
            with lock_carrinho:
                cv2.imshow(WINDOW_2, frame_carrinho)

        # ESSENCIAL para renderização
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            running = False
            break

    cv2.destroyAllWindows()
    print("[INFO] Encerrado corretamente")
