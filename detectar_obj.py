from ultralytics import YOLO
import cv2

# Carrega o modelo pré-treinado (YOLOv8 nano)
modelo = YOLO("yolov8n.pt")  # use 'yolov8s.pt', 'yolov8m.pt' etc. para versões maiores

# Carrega a imagem
imagem = "imagem.jpeg"

# Faz a detecção
resultados = modelo(imagem, show=True)  # show=True abre uma janela com o resultado

# (Opcional) Salvar imagem com os resultados
for r in resultados:
    r.save(filename="resultado.jpeg")
