import cv2
import numpy as np
import matplotlib.pyplot as plt
import configparser

def adjust_image(img, contrast=1.0, brightness=0, gamma=1.0):
    """
    Ajusta contraste, brilho e gama de uma imagem.
    pio
    :param image: Imagem de entrada (BGR).
    :param contrast: Fator de contraste (1.0 mantém o original, >1 aumenta o contraste, <1 diminui).
    :param brightness: Brilho adicional (-255 a 255).
    :param gamma: Valor de gama (1.0 mantém o original, <1 escurece, >1 clareia).
    :return: Imagem ajustada.
    """
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Ajusta contraste e brilho
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    
    # Cria uma tabela de correção de gama
    gamma_correction = np.array([
        ((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)
    ]).astype("uint8")
    
    # Aplica a correção de gama
    adjusted = cv2.LUT(adjusted, gamma_correction)
    
    return adjusted



##################################3 Ler o arquivo de configuração INI #################
config = configparser.ConfigParser()
config.read('config.ini')

# Obter os parâmetros de ajuste de imagem
contrast = float(config['ImageAdjustments']['contrast'])
brightness = int(config['ImageAdjustments']['brightness'])
gamma = float(config['ImageAdjustments']['gamma'])

# Obter as coordenadas da ROI
x = int(config['ROI']['x'])
y = int(config['ROI']['y'])
w = int(config['ROI']['width'])
h = int(config['ROI']['height'])

# Carregar a imagem do parafuso
img = cv2.imread('parafusoNOK.png')

      
gray = adjust_image(img, contrast, brightness, gamma)

# Definir a região de interesse (x, y, largura, altura)
#x, y, w, h = 156, 273, 74, 74  # Ajuste conforme necessário
#x, y, w, h = 285, 390, 55,63     # Ajuste conforme necessário


# Converter imagem em escala de cinza para BGR
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# Desenhar a ROI na imagem original
contour_img = gray_bgr  # Criar uma cópia da imagem original
# Desenhar a ROI na imagem convertida
cv2.rectangle(gray_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Desenhar ROI em azul

# Recortar a ROI da imagem em escala de cinza
roi = gray[y:y+h, x:x+w]


######################TEXTURA########################################################

# Aplicar a Transformada de Fourier- analise de textura
dft = cv2.dft(np.float32(roi), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Calcular a magnitude do espectro de frequência
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

media_textura = np.mean(magnitude_spectrum)
desvio_padrao = np.std(magnitude_spectrum)

#####################################################################################


##########  MÉDIA QUANTIFY  ############
# Calcular a média da região (Quantify)
media_roi = cv2.mean(roi)[0]

########################################

# Mostrar a imagem com ROI e contornos desenhados
print('Media da textura:',media_textura)
print('Media de quantify pixel:',media_roi)
cv2.imshow("Imagem com Contornos e ROI", gray_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()