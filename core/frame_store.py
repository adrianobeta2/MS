import threading

class FrameStore:
    def __init__(self):
        self.frames = {}
        self.lock = threading.Lock()

    def set_frame(self, cam_id, frame):
        with self.lock:
            self.frames[cam_id] = frame

    def get_frame(self, cam_id):
        with self.lock:
            return self.frames.get(cam_id, None)

    def get_all_frames(self):
        with self.lock:
            return dict(self.frames)
