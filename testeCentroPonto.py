import cv2
import numpy as np

import centroPonto


x = 455
y = 277
# Suponha que você já tenha as coordenadas do centro e o raio
center = (x, y)  # Coordenadas do centro do círculo (x, y)
radius =12   # Raio do círculo


image = cv2.imread('ref_programa1.png')

imagem, cX,cY = centroPonto.centroDoponto(image,center,radius,2,25,0.3)

print(cX,cY)