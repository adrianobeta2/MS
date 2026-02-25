import cv2

cap = cv2.VideoCapture(1)  # 0 é o índice da câmera
cap.set(cv2.CAP_PROP_BRIGHTNESS, 46)  # Ajusta o brilho
cap.set(cv2.CAP_PROP_CONTRAST, 64)    # Ajusta o contraste

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
