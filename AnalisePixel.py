
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


def pixel_analise_OK(roi,programa, gray,mask,coordenadas, pixel_tolerancia=10, contrast=1.0, brightness=0, gamma=1.0, exposure=1.0):
   
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
    status_roi = False
    best_difference = float('inf')
    best_match = None

    # Testar cada imagem de referência positiva
    for template_file in template_images_ok:
        try:
            # Carregar e processar a imagem de referência
            template_image = cv2.imread(template_file)
            if template_image is None:
                continue  # Se a imagem não existir, pula para a próxima

            gray_referencia = adjust_image_optimized_2(template_image, contrast, brightness, gamma, exposure)
            roi_ref = gray_referencia[y:y+h, x:x+w]

            if(circulo == True): 
              # Calcular a média dos pixels dentro do círculo (imagem referencia)
              media_roi_ref = float(cv2.mean(gray_referencia, mask=mask)[0])  # Converter para float
              media_roi = float(cv2.mean(gray, mask=mask)[0])  # Converter para float
              difference = abs(media_roi_ref - media_roi)
            else:
              # Calcular diferença
              media_roi_ref = float(cv2.mean(roi_ref)[0])
              media_roi = float(cv2.mean(roi)[0])
              difference = abs(media_roi_ref - media_roi)

            # Verificar se é a melhor correspondência até agora
            if difference < best_difference:
               best_difference = difference
               best_match = template_file
            
            # Se atender ao critério, marca como OK e sai do loop
            if difference < pixel_tolerancia:
               status_roi = True
               break
        except Exception as e:
         print(f"Erro ao processar {template_file}: {str(e)}")
         continue

    return status_roi,media_roi


def pixel_analise_NOK(roi,programa,gray,mask,coordenadas, pixel_tolerancia=10, contrast=1.0, brightness=0, gamma=1.0, exposure=1.0):
   
    x, y, w, h = coordenadas
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


    # Variáveis para armazenar o resultado
    status_roi = False
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
            roi_ref = gray_referencia[y:y+h, x:x+w]

            if(circulo == True): 
              # Calcular a média dos pixels dentro do círculo (imagem referencia)
              media_roi_ref = float(cv2.mean(gray_referencia, mask=mask)[0])  # Converter para float
              media_roi = float(cv2.mean(gray, mask=mask)[0])  # Converter para float
              difference = abs(media_roi_ref - media_roi)
            else:
              # Calcular diferença
              media_roi_ref = float(cv2.mean(roi_ref)[0])
              media_roi = float(cv2.mean(roi)[0])
              difference = abs(media_roi_ref - media_roi)

            # Verificar se é a melhor correspondência até agora
            if difference < best_difference:
               best_difference = difference
               best_match = template_file
            
            # Se atender ao critério, marca como OK e sai do loop
            if difference < pixel_tolerancia:
               status_roi = True
               break
        except Exception as e:
         print(f"Erro ao processar {template_file}: {str(e)}")
         continue

    return status_roi,media_roi



def new_pixel_analise_OK(roi,camera,programa, gray,mask,coordenadas, pixel_tolerancia=10, contrast=1.0, brightness=0, gamma=1.0, exposure=1.0):
   
    x, y, w, h = coordenadas
    circulo = False
    if(w==h):circulo = True

    programa =str(programa)
    camera = str(camera)
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
    status_roi = False
    best_difference = float('inf')
    best_match = None

    # Testar cada imagem de referência positiva
    for template_file in template_images_ok:
        try:
            # Carregar e processar a imagem de referência
            template_path = os.path.join(STATIC_IMAGE_FOLDER, template_file)
            template_image = cv2.imread(template_path)
            if template_image is None:
                continue  # Se a imagem não existir, pula para a próxima

            gray_referencia = adjust_image_optimized_2(template_image, contrast, brightness, gamma, exposure)
            roi_ref = gray_referencia[y:y+h, x:x+w]

            if(circulo == True): 
              # Calcular a média dos pixels dentro do círculo (imagem referencia)
              media_roi_ref = float(cv2.mean(gray_referencia, mask=mask)[0])  # Converter para float
              media_roi = float(cv2.mean(gray, mask=mask)[0])  # Converter para float
              difference = abs(media_roi_ref - media_roi)
            else:
              # Calcular diferença
              media_roi_ref = float(cv2.mean(roi_ref)[0])
              media_roi = float(cv2.mean(roi)[0])
              difference = abs(media_roi_ref - media_roi)

            # Verificar se é a melhor correspondência até agora
            if difference < best_difference:
               best_difference = difference
               best_match = template_file
            
            # Se atender ao critério, marca como OK e sai do loop
            if difference < pixel_tolerancia:
               status_roi = True
               break
        except Exception as e:
         print(f"Erro ao processar {template_file}: {str(e)}")
         continue

    return status_roi,media_roi


def new_pixel_analise_NOK(roi,camera,programa,gray,mask,coordenadas, pixel_tolerancia=10, contrast=1.0, brightness=0, gamma=1.0, exposure=1.0):
   
    x, y, w, h = coordenadas
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


    # Variáveis para armazenar o resultado
    status_roi = False
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
            roi_ref = gray_referencia[y:y+h, x:x+w]

            if(circulo == True): 
              # Calcular a média dos pixels dentro do círculo (imagem referencia)
              media_roi_ref = float(cv2.mean(gray_referencia, mask=mask)[0])  # Converter para float
              media_roi = float(cv2.mean(gray, mask=mask)[0])  # Converter para float
              difference = abs(media_roi_ref - media_roi)
            else:
              # Calcular diferença
              media_roi_ref = float(cv2.mean(roi_ref)[0])
              media_roi = float(cv2.mean(roi)[0])
              difference = abs(media_roi_ref - media_roi)

            # Verificar se é a melhor correspondência até agora
            if difference < best_difference:
               best_difference = difference
               best_match = template_file
            
            # Se atender ao critério, marca como OK e sai do loop
            if difference < pixel_tolerancia:
               status_roi = True
               break
        except Exception as e:
         print(f"Erro ao processar {template_file}: {str(e)}")
         continue

    return status_roi,media_roi