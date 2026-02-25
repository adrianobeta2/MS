import cv2
import numpy as np

x = 292
y = 173
# Suponha que você já tenha as coordenadas do centro e o raio
center = (x, y)  # Coordenadas do centro do círculo (x, y)
radius =113    # Raio do círculo

reference_colors = []

# Carregar a imagem (substitua 'template_image' pela sua imagem)
template_image = cv2.imread('ref_programa1.png')

# Converter a imagem para HSV
hsv_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2HSV)

# Inicializar uma lista para armazenar os valores de cor dentro do círculo
pixels_dentro_do_circulo = []

# Iterar sobre a região retangular que contém o círculo
for i in range(y - radius, y + radius + 1):          # Linhas (eixo y)
    for j in range(x - radius, x + radius + 1):      # Colunas (eixo x)
        # Verificar se o pixel (j, i) está dentro do círculo
        if (j - x) ** 2 + (i - y) ** 2 <= radius ** 2:
            # Adicionar o valor do pixel à lista
            pixels_dentro_do_circulo.append(hsv_image[i, j])

# Calcular a média de cor apenas dos pixels dentro do círculo
if pixels_dentro_do_circulo:  # Verifica se a lista não está vazia
    mean_color_ref = np.mean(pixels_dentro_do_circulo, axis=0)
else:
    mean_color_ref = None  # Caso não haja pixels dentro do círculo

# Adicionar a média de cor à lista de referências
reference_colors.append(mean_color_ref)

print("Cor média dentro do círculo:", mean_color_ref)