import cv2
import numpy as np
import configparser


# Ler o arquivo de configuração INI
config = configparser.ConfigParser()
config.read('config.ini')

def adjust_image_optimized(img, contrast=1.9, brightness=30, gamma=0.5):
    """
    Ajusta contraste, brilho e gama de uma imagem em um único passo.
    """
    # Converter para escala de cinza e ajustar contraste/brilho
    adjusted = cv2.convertScaleAbs(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), alpha=contrast, beta=brightness)

     # Aplicar equalização de histograma para realçar contraste
    # adjusted = cv2.equalizeHist(adjusted)
    
    # Aplicar correção de gama (tabela de lookup otimizada)
    gamma_correction = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype="uint8")
    return cv2.LUT(adjusted, gamma_correction)


def adjust_image_optimized_2(img, contrast=1.0, brightness=0, gamma=1.0, exposure=1.0):
    """
    Ajusta contraste, brilho, gama e exposição de uma imagem em um único passo.
    
    Args:
        img: Imagem de entrada (BGR).
        contrast: Fator de contraste (default: 1.0).
        brightness: Valor de brilho (default: 0).
        gamma: Valor de correção de gama (default: 1.0).
        exposure: Fator de exposição (default: 1.0).
    Returns:
        Imagem ajustada em escala de cinza.
    """
    # Converter para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar exposição (ajustando brilho e contraste)
    adjusted = cv2.convertScaleAbs(gray, alpha=contrast * exposure, beta=brightness * exposure)
    
    # Aplicar nitidez (unsharp masking)
    blurred = cv2.GaussianBlur(adjusted, (5, 5), 0)
    adjusted = cv2.addWeighted(adjusted, 1.5, blurred, -0.5, 0)
    
    # Aplicar correção de gama (tabela de lookup otimizada)
    gamma_correction = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype="uint8")
    return cv2.LUT(adjusted, gamma_correction)


def centroDoponto(image, roi_center, roi_radius, contrast,brightness, gamma,exposure, tolerance=150):

    gray = adjust_image_optimized_2(image, contrast, brightness, gamma, exposure)
    # Aplicar um filtro Gaussiano para reduzir ruído
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Criar uma máscara circular
    mask = np.zeros_like(gray)  # Máscara preta (tudo zero)
    cv2.circle(mask, roi_center, roi_radius, 255, -1)  # Desenhar um círculo branco na máscara

    # Aplicar a máscara à imagem
    masked_image = cv2.bitwise_and(blurred, blurred, mask=mask)

    # Encontrar o ponto mais escuro dentro da ROI
    min_val, _, min_loc, _ = cv2.minMaxLoc(masked_image, mask=mask)

    # Verificar se o ponto mais escuro foi encontrado
    if min_val < 100:  # Ajuste o valor conforme necessário
        cX, cY = min_loc  # Coordenadas do ponto mais escuro
        
        # Desenhar o centro do ponto preto  
        cv2.circle(image, (cX, cY), 3, (255, 0, 0), -1)
        
        # Exibir as coordenadas do centro
        print(f"Centro do ponto preto dentro da ROI: ({cX}, {cY})")
    else:

        cX = 0
        cY = 0
        print("Nenhum ponto escuro detectado dentro da ROI.")

    

    return image, cX, cY, min_val