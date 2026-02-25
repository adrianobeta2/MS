
from flask import Flask,abort, request, Response, jsonify, send_file,stream_with_context, render_template
import configparser
import cv2
import threading
import os
from flask_cors import CORS
import requests
import json
import numpy as np
import time
from datetime import datetime
import centroPonto
import Localizador
import AnaliseCor
import angulo
from pypylon import pylon


STATIC_IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'imagens')


def adjust_image_optimized_2(img, contrast=1.0, brightness=0, gamma=1.0, exposure=1.0):
    
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


def textura_analise_OK(roi,programa, gray,mask,coordenadas, tolerancia=10, contrast=1.0, brightness=0, gamma=1.0, exposure=1.0):
   
    x, y, w, h = coordenadas
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

    # Variáveis para armazenar o resultado
    status = False
    

    # Testar cada imagem de referência positiva
    for template_file in template_images_ok:
        try:
            # Carregar e processar a imagem de referência
            template_image = cv2.imread(template_file)
            if template_image is None:
                continue  # Se a imagem não existir, pula para a próxima

            gray_referencia = adjust_image_optimized_2(template_image, contrast, brightness, gamma, exposure)
           

            if(circulo == True): 
              # Criar uma máscara preta do tamanho da imagem
              mask = np.zeros_like(gray, dtype=np.uint8)
              # Desenhar um círculo branco na máscara
              cv2.circle(mask, (x, y), w, 255, -1)
              # Aplicar a máscara na imagem original para obter apenas a região circular
              roi_ref = cv2.bitwise_and(gray_referencia, gray_referencia, mask=mask) #imagem referencia
              roi = cv2.bitwise_and(gray, gray, mask=mask)

            else:
              mask = None
              # Recortar a ROI
              roi_ref = gray_referencia[y:y+h, x:x+w] #imagem referencia
              roi = gray[y:y+h, x:x+w]
            
            dft = cv2.dft(np.float32(roi_ref), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
            media_textura_ref = float(np.mean(magnitude_spectrum))  # Converter para float

            dft = cv2.dft(np.float32(roi), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
            media_textura = float(np.mean(magnitude_spectrum))  # Converter para float
            
            
            difference = abs(media_textura_ref- media_textura)
            
            # Se atender ao critério, marca como OK e sai do loop
            if difference < tolerancia:
               status = True
               break
        except Exception as e:
         print(f"Erro ao processar {template_file}: {str(e)}")
         continue

    return status,media_textura


def textura_analise_NOK(roi,programa, gray,mask,coordenadas, tolerancia=10, contrast=1.0, brightness=0, gamma=1.0, exposure=1.0):
   
    x, y, w, h = coordenadas
    circulo = False
    if(w==h):circulo = True

    programa =str(programa)
    # Lista de imagens de referência positiva (adicionar mais conforme necessário)
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
        # adicione mais se necessário
    ]

    # Variáveis para armazenar o resultado
    status = False
    best_difference = float('inf')
    best_match = None

    # Testar cada imagem de referência positiva
    for template_file in template_images_nok:
        try:
            # Carregar e processar a imagem de referência
            template_image = cv2.imread(template_file)
            if template_image is None:
                continue  # Se a imagem não existir, pula para a próxima

            gray_referencia = adjust_image_optimized_2(template_image, contrast, brightness, gamma, exposure)
           

            if(circulo == True): 
              # Criar uma máscara preta do tamanho da imagem
              mask = np.zeros_like(gray, dtype=np.uint8)
              # Desenhar um círculo branco na máscara
              cv2.circle(mask, (x, y), w, 255, -1)
              # Aplicar a máscara na imagem original para obter apenas a região circular
              roi_ref = cv2.bitwise_and(gray_referencia, gray_referencia, mask=mask) #imagem referencia
              roi = cv2.bitwise_and(gray, gray, mask=mask)

            else:
              mask = None
              # Recortar a ROI
              roi_ref = gray_referencia[y:y+h, x:x+w] #imagem referencia
              roi = gray[y:y+h, x:x+w]
            
            dft = cv2.dft(np.float32(roi_ref), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
            media_textura_ref = float(np.mean(magnitude_spectrum))  # Converter para float

            dft = cv2.dft(np.float32(roi), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
            media_textura = float(np.mean(magnitude_spectrum))  # Converter para float
            
            
            difference = abs(media_textura_ref- media_textura)
            
            # Se atender ao critério, marca como OK e sai do loop
            if difference < tolerancia:
               status = True
               break
        except Exception as e:
         print(f"Erro ao processar {template_file}: {str(e)}")
         continue

    return status,media_textura


def new_textura_analise_OK(roi,camera,programa, gray,mask,coordenadas, tolerancia=10, contrast=1.0, brightness=0, gamma=1.0, exposure=1.0):
   
    x, y, w, h = coordenadas
    circulo = False
    if(w==h):circulo = True
    camera = str(camera)
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

    # Variáveis para armazenar o resultado
    status = False
    

    # Testar cada imagem de referência positiva
    for template_file in template_images_ok:
        try:
            # Carregar e processar a imagem de referência
            template_path = os.path.join(STATIC_IMAGE_FOLDER, template_file)
            template_image = cv2.imread(template_path)
            if template_image is None:
                continue  # Se a imagem não existir, pula para a próxima

            gray_referencia = adjust_image_optimized_2(template_image, contrast, brightness, gamma, exposure)
           

            if(circulo == True): 
              # Criar uma máscara preta do tamanho da imagem
              mask = np.zeros_like(gray, dtype=np.uint8)
              # Desenhar um círculo branco na máscara
              cv2.circle(mask, (x, y), w, 255, -1)
              # Aplicar a máscara na imagem original para obter apenas a região circular
              roi_ref = cv2.bitwise_and(gray_referencia, gray_referencia, mask=mask) #imagem referencia
              roi = cv2.bitwise_and(gray, gray, mask=mask)

            else:
              mask = None
              # Recortar a ROI
              roi_ref = gray_referencia[y:y+h, x:x+w] #imagem referencia
              roi = gray[y:y+h, x:x+w]
            
            dft = cv2.dft(np.float32(roi_ref), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
            media_textura_ref = float(np.mean(magnitude_spectrum))  # Converter para float

            dft = cv2.dft(np.float32(roi), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
            media_textura = float(np.mean(magnitude_spectrum))  # Converter para float
            
            
            difference = abs(media_textura_ref- media_textura)
            
            # Se atender ao critério, marca como OK e sai do loop
            if difference < tolerancia:
               status = True
               break
        except Exception as e:
         print(f"Erro ao processar {template_file}: {str(e)}")
         continue

    return status,media_textura


def new_textura_analise_NOK(roi,camera,programa, gray,mask,coordenadas, tolerancia=10, contrast=1.0, brightness=0, gamma=1.0, exposure=1.0):
   
    x, y, w, h = coordenadas
    circulo = False
    if(w==h):circulo = True
    
    camera = str(camera)
    programa =str(programa)
    # Lista de imagens de referência positiva (adicionar mais conforme necessário)
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
        # adicione mais se necessário
    ]

    # Variáveis para armazenar o resultado
    status = False
    best_difference = float('inf')
    best_match = None

    # Testar cada imagem de referência positiva
    for template_file in template_images_nok:
        try:
            # Carregar e processar a imagem de referência
            template_path = os.path.join(STATIC_IMAGE_FOLDER, template_file)
            template_image = cv2.imread(template_path)
            if template_image is None:
                continue  # Se a imagem não existir, pula para a próxima

            gray_referencia = adjust_image_optimized_2(template_image, contrast, brightness, gamma, exposure)
           

            if(circulo == True): 
              # Criar uma máscara preta do tamanho da imagem
              mask = np.zeros_like(gray, dtype=np.uint8)
              # Desenhar um círculo branco na máscara
              cv2.circle(mask, (x, y), w, 255, -1)
              # Aplicar a máscara na imagem original para obter apenas a região circular
              roi_ref = cv2.bitwise_and(gray_referencia, gray_referencia, mask=mask) #imagem referencia
              roi = cv2.bitwise_and(gray, gray, mask=mask)

            else:
              mask = None
              # Recortar a ROI
              roi_ref = gray_referencia[y:y+h, x:x+w] #imagem referencia
              roi = gray[y:y+h, x:x+w]
            
            dft = cv2.dft(np.float32(roi_ref), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
            media_textura_ref = float(np.mean(magnitude_spectrum))  # Converter para float

            dft = cv2.dft(np.float32(roi), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
            media_textura = float(np.mean(magnitude_spectrum))  # Converter para float
            
            
            difference = abs(media_textura_ref- media_textura)
            
            # Se atender ao critério, marca como OK e sai do loop
            if difference < tolerancia:
               status = True
               break
        except Exception as e:
         print(f"Erro ao processar {template_file}: {str(e)}")
         continue

    return status,media_textura



