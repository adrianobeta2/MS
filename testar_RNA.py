import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suprime INFO e WARNING
import tensorflow as tf


# Pasta das imagens
IMG_DIR = os.path.join(os.path.dirname(__file__), 'static', 'imagens')

def carregar_imagem(caminho, coordenadas):
    """Carrega imagem em grayscale, recorta ROI e redimensiona"""
    x, y, w, h = coordenadas
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    roi = img[y:y+h, x:x+w]
    roi = cv2.resize(roi, (28,28))
    roi = roi.astype("float32") / 255.0
    return roi.reshape(1,28,28,1)

def predizer_frame(model_path, frame_path, coordenadas):
    """
    Carrega modelo e prediz se frame está OK (1) ou NOK (0)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if not os.path.exists(frame_path):
        raise FileNotFoundError(f"Frame não encontrado: {frame_path}")

    model = load_model(model_path)
    img = carregar_imagem(frame_path, coordenadas)
    pred = model.predict(img)
    resultado = 1 if pred[0][0] >= 0.5 else 0
    return resultado

# Exemplo de uso
if __name__ == "__main__":
    coordenadas = (361, 316, 12, 12)
    model_path = os.path.join(os.path.dirname(__file__), "modelo_cam1_prog1_ROI1.keras")
    frame_path = os.path.join(IMG_DIR, "cam1_frame_teste.png")

    resultado = predizer_frame(model_path, frame_path, coordenadas)
    print(f"Predição do frame teste: {'OK' if resultado==1 else 'NOK'}")
