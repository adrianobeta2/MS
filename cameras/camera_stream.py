import cv2
import threading
import time

class CameraStream(threading.Thread):
    def __init__(self, cam_id, src, frame_store, width=None, height=None):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.src = src
        self.frame_store = frame_store
        self.running = True

        self.cap = cv2.VideoCapture(src)

        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError(f"Erro ao abrir câmera {cam_id}")

    def run(self):
        print(f"[CAM {self.cam_id}] Thread iniciada")
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_store.set_frame(self.cam_id, frame)
            else:
                time.sleep(0.05)

    def stop(self):
        self.running = False
        self.cap.release()
