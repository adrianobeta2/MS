import cv2
import numpy as np
import configparser
import snap 
import time
import requests

def adjust_image_optimized(img, contrast=1.0, brightness=0, gamma=1.0):
    """
    Ajusta contraste, brilho e gama de uma imagem em um único passo.
    """
    # Converter para escala de cinza e ajustar contraste/brilho
    adjusted = cv2.convertScaleAbs(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), alpha=contrast, beta=brightness)
    
    # Aplicar correção de gama (tabela de lookup otimizada)
    gamma_correction = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype="uint8")
    return cv2.LUT(adjusted, gamma_correction)

def adjust_image(img, contrast=1.0, brightness=0, gamma=1.0):
    """
    Ajusta contraste, brilho e gama de uma imagem.
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

# Ler o arquivo de configuração INI
config = configparser.ConfigParser()
config.read('config.ini')

# Obter os parâmetros de ajuste de imagem
contrast = float(config['ImageAdjustments']['contrast'])
brightness = int(config['ImageAdjustments']['brightness'])
gamma = float(config['ImageAdjustments']['gamma'])

# Obter as coordenadas das ROIs
# Validar a existência das ROIs no arquivo de configuração
rois = []
for i in range(1, 4):  # Três ROIs
    section_name = f'ROI{i}'
    if section_name in config:
        try:
            x = int(config[section_name]['x'])
            y = int(config[section_name]['y'])
            w = int(config[section_name]['width'])
            h = int(config[section_name]['height'])
            rois.append((x, y, w, h))
        except KeyError as e:
            print(f"Erro: A chave {e} está ausente na seção {section_name}. Verifique o arquivo config.ini.")
            exit(1)
    else:
        print(f"Erro: Seção {section_name} não encontrada no arquivo config.ini.")
        exit(1)




#url = "http://192.168.1.100:6001/capture"
url = "http://127.0.0.1:6001/capture"

response = requests.get(url)
if response.status_code == 200:
    # Decodificar o frame JPEG recebido
    nparr = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

   
else:
    print(f"Erro ao capturar frame: {response.json()['error']}")



# Ajustar a imagem
gray = adjust_image_optimized(img, contrast, brightness, gamma)

# Converter imagem em escala de cinza para BGR
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# Processar cada ROI
for i, (x, y, w, h) in enumerate(rois, start=1):
    # Desenhar a ROI na imagem
    cv2.rectangle(gray_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Recortar a ROI
    roi = gray[y:y+h, x:x+w]
    
    # Análise de textura
    dft = cv2.dft(np.float32(roi), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    
    media_textura = np.mean(magnitude_spectrum)
    desvio_padrao = np.std(magnitude_spectrum)

    ##########  MÉDIA QUANTIFY  ############
    # Calcular a média da região (Quantify)
    media_roi = cv2.mean(roi)[0]


    

    
    
    # Exibir todas as medições
    print(f'ROI {i}:')
    print(f'  Valores de textura media: {media_textura}')
    print(f'  Valores de pixels na ROI: {media_roi}')
    print()

# Salvar a imagem com ROIs desenhadas em um arquivo JPEG
cv2.imwrite('imagem_com_rois.jpg', gray_bgr)
# Mostrar a imagem com ROIs desenhadas
cv2.imshow("Imagem com ROIs", gray_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
