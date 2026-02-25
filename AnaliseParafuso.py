import cv2
import numpy as np
import configparser
import os

STATIC_IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'imagens')

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


def ParafusoAnaliseOK(image,programa, coordenadas, contrast,brightness, gamma,exposure, tolerance=150):

    x, y, w, h = coordenadas
  
    roi_center = (x,y)
    roi_radius = w
    circulo = False
    if(w==h):circulo = True
    programa =str(programa)
    
     # Lista de imagens de referência positiva (adicionar mais conforme necessário)
    template_images_ok = [
        'ref_programa'+programa+'_OK.png',
        'ref_programa'+programa+'_OK_1.png',
        'ref_programa'+programa+'_OK_2.png',
        'ref_programa'+programa+'_OK_3.png',
        'ref_programa'+programa+'_OK_4.png',
        'ref_programa'+programa+'_OK_5.png',
        'ref_programa'+programa+'_OK_6.png',
        'ref_programa'+programa+'_OK_7.png',
        'ref_programa'+programa+'_OK_8.png',
        'ref_programa'+programa+'_OK_9.png',
        'ref_programa'+programa+'_OK_10.png'
        # adicione mais se necessário
    ]
    status = False
    # Testar cada imagem de referência positiva
    for template_file in template_images_ok:
        try:   
                cX = 0
                cY = 0
                # Carregar e processar a imagem de referência
                template_image = cv2.imread(template_file)
                if template_image is None:
                   continue  # Se a imagem não existir, pula para a próxima
                ############################# Analise referencia ###################
                gray_ref = adjust_image_optimized_2(template_image, contrast, brightness, gamma, exposure)
                # Aplicar um filtro Gaussiano para reduzir ruído
                blurred = cv2.GaussianBlur(gray_ref, (5, 5), 0)
                # Criar uma máscara circular
                mask = np.zeros_like(gray_ref)  # Máscara preta (tudo zero)
                cv2.circle(mask, roi_center, roi_radius, 255, -1)  # Desenhar um círculo branco na máscara

                # Aplicar a máscara à imagem
                masked_image_ref = cv2.bitwise_and(blurred, blurred, mask=mask)

                # Encontrar o ponto mais escuro dentro da ROI
                min_val_reference, _, min_loc, _ = cv2.minMaxLoc(masked_image_ref, mask=mask)

                ############################# Analise imagem atual ###################
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


                difference = abs(min_val_reference - min_val)
                if difference < tolerance:
                   status = True
                   break
                else:
                   status = False   
                   cX, cY = min_loc  # Coordenadas do ponto mais escuro      
                   

                # Verificar se o ponto mais escuro foi encontrado
                if min_val < 100:  # Ajuste o valor conforme necessário
                    cX, cY = min_loc  # Coordenadas do ponto mais escuro
                    
                    # Desenhar o centro do ponto preto  
                    cv2.circle(image, (cX, cY), 3, (255, 0, 0), -1)
                    
                    # Exibir as coordenadas do centro
                    print(f"Centro do ponto preto dentro da ROI: ({cX}, {cY})")
                  
        except Exception as e:
         print(f"Erro ao processar {template_file}: {str(e)}")
         continue
    

    return cX, cY, status



def ParafusoAnaliseNOK(image,programa, coordenadas, contrast,brightness, gamma,exposure, tolerance=150):

    x, y, w, h = coordenadas
  
    roi_center = (x,y)
    roi_radius = w
    circulo = False
    if(w==h):circulo = True
    programa =str(programa)
    
     # Lista de imagens de referência negativa (adicionar mais conforme necessário)
    template_images_nok = [
        'ref_programa'+programa+'_NOK.png',
        'ref_programa'+programa+'_NOK_1.png',
        'ref_programa'+programa+'_NOK_2.png',
        'ref_programa'+programa+'_NOK_3.png',
        'ref_programa'+programa+'_NOK_4.png',
        'ref_programa'+programa+'_NOK_5.png',
        'ref_programa'+programa+'_NOK_6.png',
        'ref_programa'+programa+'_NOK_7.png',
        'ref_programa'+programa+'_NOK_8.png',
        'ref_programa'+programa+'_NOK_9.png',
        'ref_programa'+programa+'_NOK_10.png'
        
    ]
    status = False
    # Testar cada imagem de referência positiva
    for template_file in template_images_nok:
        try:
                cX = 0
                cY = 0
                # Carregar e processar a imagem de referência
                template_image = cv2.imread(template_file)
                if template_image is None:
                   continue  # Se a imagem não existir, pula para a próxima
                ############################# Analise referencia ###################
                gray_ref = adjust_image_optimized_2(template_image, contrast, brightness, gamma, exposure)
                # Aplicar um filtro Gaussiano para reduzir ruído
                blurred = cv2.GaussianBlur(gray_ref, (5, 5), 0)
                # Criar uma máscara circular
                mask = np.zeros_like(gray_ref)  # Máscara preta (tudo zero)
                cv2.circle(mask, roi_center, roi_radius, 255, -1)  # Desenhar um círculo branco na máscara

                # Aplicar a máscara à imagem
                masked_image_ref = cv2.bitwise_and(blurred, blurred, mask=mask)

                # Encontrar o ponto mais escuro dentro da ROI
                min_val_reference, _, min_loc, _ = cv2.minMaxLoc(masked_image_ref, mask=mask)

                ############################# Analise imagem atual ###################
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


                difference = abs(min_val_reference - min_val)
                if difference < tolerance:
                   status = True
                   cX, cY = min_loc  # Coordenadas do ponto mais escuro
                   break
                else:
                   status = False   
                  
                   

                # Verificar se o ponto mais escuro foi encontrado
                if min_val < 100:  # Ajuste o valor conforme necessário
                    cX, cY = min_loc  # Coordenadas do ponto mais escuro
                    
                    # Desenhar o centro do ponto preto  
                    cv2.circle(image, (cX, cY), 3, (255, 0, 0), -1)
                    
                    # Exibir as coordenadas do centro
                    print(f"Centro do ponto preto dentro da ROI: ({cX}, {cY})")
                  
        except Exception as e:
         print(f"Erro ao processar {template_file}: {str(e)}")
         continue
    

    return cX, cY, status


def new_ParafusoAnaliseOK(image,camera,programa, coordenadas, contrast,brightness, gamma,exposure, tolerance=150):

    x, y, w, h = coordenadas
  
    roi_center = (x,y)
    roi_radius = w
    circulo = False
    if(w==h):circulo = True
    
    camera = str(programa)
    programa =str(programa)
    
     # Lista de imagens de referência positiva (adicionar mais conforme necessário)
    template_images_ok = [
        'cam'+camera+'_ref_programa'+programa+'_OK.png',
        'cam'+camera+'_ref_programa'+programa+'_OK_1.png',
        'cam'+camera+'_ref_programa'+programa+'_OK_2.png',
        'cam'+camera+'_ref_programa'+programa+'_OK_3.png',
        'cam'+camera+'_ref_programa'+programa+'_OK_4.png',
        'cam'+camera+'_ref_programa'+programa+'_OK_5.png',
        'cam'+camera+'_ref_programa'+programa+'_OK_6.png',
        'cam'+camera+'_ref_programa'+programa+'_OK_7.png',
        'cam'+camera+'_ref_programa'+programa+'_OK_8.png',
        'cam'+camera+'_ref_programa'+programa+'_OK_9.png',
        'cam'+camera+'_ref_programa'+programa+'_OK_10.png'
        # adicione mais se necessário
    ]
    status = False
    # Testar cada imagem de referência positiva
    for template_file in template_images_ok:
        try:   
                cX = 0
                cY = 0
                # Carregar e processar a imagem de referência
                template_path = os.path.join(STATIC_IMAGE_FOLDER, template_file)
                template_image = cv2.imread(template_path)
                if template_image is None:
                   continue  # Se a imagem não existir, pula para a próxima
                ############################# Analise referencia ###################
                gray_ref = adjust_image_optimized_2(template_image, contrast, brightness, gamma, exposure)
                # Aplicar um filtro Gaussiano para reduzir ruído
                blurred = cv2.GaussianBlur(gray_ref, (5, 5), 0)
                # Criar uma máscara circular
                mask = np.zeros_like(gray_ref)  # Máscara preta (tudo zero)
                cv2.circle(mask, roi_center, roi_radius, 255, -1)  # Desenhar um círculo branco na máscara

                # Aplicar a máscara à imagem
                masked_image_ref = cv2.bitwise_and(blurred, blurred, mask=mask)

                # Encontrar o ponto mais escuro dentro da ROI
                min_val_reference, _, min_loc, _ = cv2.minMaxLoc(masked_image_ref, mask=mask)

                ############################# Analise imagem atual ###################
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


                difference = abs(min_val_reference - min_val)
                if difference < tolerance:
                   status = True
                   break
                else:
                   status = False   
                   cX, cY = min_loc  # Coordenadas do ponto mais escuro      
                   

                # Verificar se o ponto mais escuro foi encontrado
                if min_val < 100:  # Ajuste o valor conforme necessário
                    cX, cY = min_loc  # Coordenadas do ponto mais escuro
                    
                    # Desenhar o centro do ponto preto  
                    cv2.circle(image, (cX, cY), 3, (255, 0, 0), -1)
                    
                    # Exibir as coordenadas do centro
                    print(f"Centro do ponto preto dentro da ROI: ({cX}, {cY})")
                  
        except Exception as e:
         print(f"Erro ao processar {template_file}: {str(e)}")
         continue
    

    return cX, cY, status



def new_ParafusoAnaliseNOK(image,camera,programa, coordenadas, contrast,brightness, gamma,exposure, tolerance=150):

    x, y, w, h = coordenadas
  
    roi_center = (x,y)
    roi_radius = w
    circulo = False
    if(w==h):circulo = True
    camera = str(camera)
    programa =str(programa)
    
     # Lista de imagens de referência negativa (adicionar mais conforme necessário)
    template_images_nok = [
        'cam'+camera+'_ref_programa'+programa+'_NOK.png',
        'cam'+camera+'_ref_programa'+programa+'_NOK_1.png',
        'cam'+camera+'_ref_programa'+programa+'_NOK_2.png',
        'cam'+camera+'_ref_programa'+programa+'_NOK_3.png',
        'cam'+camera+'_ref_programa'+programa+'_NOK_4.png',
        'cam'+camera+'_ref_programa'+programa+'_NOK_5.png',
        'cam'+camera+'_ref_programa'+programa+'_NOK_6.png',
        'cam'+camera+'_ref_programa'+programa+'_NOK_7.png',
        'cam'+camera+'_ref_programa'+programa+'_NOK_8.png',
        'cam'+camera+'_ref_programa'+programa+'_NOK_9.png',
        'cam'+camera+'_ref_programa'+programa+'_NOK_10.png'
        
    ]
    status = False
    # Testar cada imagem de referência positiva
    for template_file in template_images_nok:
        try:
                cX = 0
                cY = 0
                # Carregar e processar a imagem de referência
                template_path = os.path.join(STATIC_IMAGE_FOLDER, template_file)
                template_image = cv2.imread(template_path)
                if template_image is None:
                   continue  # Se a imagem não existir, pula para a próxima
                ############################# Analise referencia ###################
                gray_ref = adjust_image_optimized_2(template_image, contrast, brightness, gamma, exposure)
                # Aplicar um filtro Gaussiano para reduzir ruído
                blurred = cv2.GaussianBlur(gray_ref, (5, 5), 0)
                # Criar uma máscara circular
                mask = np.zeros_like(gray_ref)  # Máscara preta (tudo zero)
                cv2.circle(mask, roi_center, roi_radius, 255, -1)  # Desenhar um círculo branco na máscara

                # Aplicar a máscara à imagem
                masked_image_ref = cv2.bitwise_and(blurred, blurred, mask=mask)

                # Encontrar o ponto mais escuro dentro da ROI
                min_val_reference, _, min_loc, _ = cv2.minMaxLoc(masked_image_ref, mask=mask)

                ############################# Analise imagem atual ###################
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


                difference = abs(min_val_reference - min_val)
                if difference < tolerance:
                   status = True
                   cX, cY = min_loc  # Coordenadas do ponto mais escuro
                   break
                else:
                   status = False   
                  
                   

                # Verificar se o ponto mais escuro foi encontrado
                if min_val < 100:  # Ajuste o valor conforme necessário
                    cX, cY = min_loc  # Coordenadas do ponto mais escuro
                    
                    # Desenhar o centro do ponto preto  
                    cv2.circle(image, (cX, cY), 3, (255, 0, 0), -1)
                    
                    # Exibir as coordenadas do centro
                    print(f"Centro do ponto preto dentro da ROI: ({cX}, {cY})")
                  
        except Exception as e:
         print(f"Erro ao processar {template_file}: {str(e)}")
         continue
    

    return cX, cY, status