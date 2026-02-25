import time
import cv2

from core.frame_store import FrameStore
from cameras.camera_stream import CameraStream

# Buffer compartilhado
frame_store = FrameStore()

# Fontes das câmeras (USB, RTSP, IP, etc)
CAMERAS = {
    "cam1": 0,
    "cam2": 2
    #"cam3": "rtsp://usuario:senha@ip:554/stream1"
}

streams = []

# Inicia threads
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

print("🎥 Todas as câmeras iniciadas")

# Exemplo: consumo dos frames em outro loop
try:
    while True:
        frame = frame_store.get_frame("cam1")
        if frame is not None:
            cv2.imshow("Cam 1", frame)

        if cv2.waitKey(1) == 27:
            break

        time.sleep(0.01)

except KeyboardInterrupt:
    pass

finally:
    for s in streams:
        s.stop()
    cv2.destroyAllWindows()
