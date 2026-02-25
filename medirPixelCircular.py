import cv2
import numpy as np
x = 292
y = 173
# Suponha que você já tenha as coordenadas do centro e o raio
center = (x, y)  # Coordenadas do centro do círculo (x, y)
radius =113    # Raio do círculo
# Carregar a imagem em escala de cinza (substitua 'gray' pela sua imagem)
gray = cv2.imread('ref_programa1.png', cv2.IMREAD_GRAYSCALE)

# Inicializar uma lista para armazenar os valores dos pixels dentro do círculo
pixels_dentro_do_circulo = []

# Iterar sobre uma região retangular que contém o círculo
for i in range(y - radius, y + radius + 1):          # Linhas (eixo y)
    for j in range(x - radius, x + radius + 1):      # Colunas (eixo x)
        # Verificar se o pixel (j, i) está dentro do círculo
        if (j - x) ** 2 + (i - y) ** 2 <= radius ** 2:
            # Adicionar o valor do pixel à lista
            pixels_dentro_do_circulo.append(gray[i, j])

# Converter a lista para um array numpy (opcional)
pixels_dentro_do_circulo = np.array(pixels_dentro_do_circulo)

# Calcular a média dos pixels dentro do círculo (opcional)
media_pixels = np.mean(pixels_dentro_do_circulo)
print("Média dos pixels dentro do círculo:", media_pixels)

# Se quiser criar uma imagem apenas com a ROI circular (opcional)
roi_circular = np.zeros_like(gray)  # Imagem preta do mesmo tamanho da original
for i in range(y - radius, y + radius + 1):
    for j in range(x - radius, x + radius + 1):
        if (j - x) ** 2 + (i - y) ** 2 <= radius ** 2:
            roi_circular[i, j] = gray[i, j]  # Copiar os pixels dentro do círculo

# Exibir a ROI circular
cv2.imshow('ROI Circular sem Máscara', roi_circular)
cv2.waitKey(0)
cv2.destroyAllWindows()