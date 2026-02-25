import cv2
from facial_gpu import reconhecer_api

VIDEO_PATH = "saida2.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated = reconhecer_api(frame)
    cv2.imshow("GPU Facial", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
