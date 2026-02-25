import cv2
from ultralytics import YOLO

# Carrega o modelo pré-treinado (YOLOv8 nano - mais rápido)
modelo = YOLO("yolov8n.pt")

# Inicia a captura da webcam (0 = webcam padrão)
cap = cv2.VideoCapture(0)

# Verifica se a webcam abriu corretamente
if not cap.isOpened():
    print("Erro ao acessar a webcam")
    exit()

# Loop de captura
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar imagem")
        break

    # Realiza a detecção no frame da webcam
    resultados = modelo.predict(source=frame, show=False, conf=0.5)

    # Converte resultados para imagem com anotações
    anotado = resultados[0].plot()  # desenha as caixas no frame

    # Mostra o resultado
    cv2.imshow("YOLOv8 - Webcam", anotado)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()
