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
import angulo
from pypylon import pylon


STATIC_IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'imagens')



# Função para calcular a diferença entre duas cores (em HSV)
def color_difference(hsv1, hsv2):
    return np.linalg.norm(np.array(hsv1) - np.array(hsv2))

def Cor(image,programa,cor_tolerancia, coordenadas ):
     circulo = False
     x,y,w,h = coordenadas
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
     status_cor = False

     # Testar cada imagem de referência positiva
     for template_file in template_images_ok:
        try:
              # Carregar e processar a imagem de referência
              template_image = cv2.imread(template_file)
              if circulo:
                     center_x, center_y, radius = x, y, w 


                     # Criar uma máscara do mesmo tamanho da imagem referencia
                     mask = np.zeros(template_image.shape[:2], dtype=np.uint8)
                     # Desenhar um círculo branco na máscara
                     cv2.circle(mask, (center_x, center_y), radius, 255, -1)
                     # Aplicar a máscara na imagem
                     hsv_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2HSV)
                     hsv_roi_ref = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
                     # Calcular a média apenas dos pixels dentro do círculo
                     mean_ref = cv2.mean(hsv_roi_ref, mask=mask)[:3]

                     # Criar uma máscara do mesmo tamanho da imagem da peça
                     mask = np.zeros(image.shape[:2], dtype=np.uint8)
                     # Desenhar um círculo branco na máscara
                     cv2.circle(mask, (center_x, center_y), radius, 255, -1)
                     # Aplicar a máscara na imagem
                     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                     hsv_roi_ref = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
                     # Calcular a média apenas dos pixels dentro do círculo
                     mean = cv2.mean(hsv_roi_ref, mask=mask)[:3]

                     difference = color_difference(mean_ref,mean)


                     # Armazenar os valores de referência
                     #reference_colors.append(mean_color_ref)
              else: 
                     roi_ref = template_image[y:y+h, x:x+w]
                     hsv_roi_ref = cv2.cvtColor(roi_ref, cv2.COLOR_BGR2HSV)
                     mean_ref = cv2.mean(hsv_roi_ref)[:3]

                     roi_ref = image[y:y+h, x:x+w]
                     hsv_roi_ref = cv2.cvtColor(roi_ref, cv2.COLOR_BGR2HSV)
                     mean = cv2.mean(hsv_roi_ref)[:3] 
                     
                     difference = color_difference(mean_ref,mean)
              
              # Se atender ao critério, marca como OK e sai do loop
              if difference < cor_tolerancia:
                     status_cor = True
                     break


                     

        except Exception as e:
         print(f"Erro ao processar {template_file}: {str(e)}")
         continue
     return status_cor

def Cor_new(image,camera,programa,cor_tolerancia, coordenadas ):
     circulo = False
     x,y,w,h = coordenadas
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
     status_cor = False

     # Testar cada imagem de referência positiva
     for template_file in template_images_ok:
        try:
              template_path = os.path.join(STATIC_IMAGE_FOLDER, template_file)
              template_image = cv2.imread(template_path)
             
              if circulo:
                     center_x, center_y, radius = x, y, w 


                     # Criar uma máscara do mesmo tamanho da imagem referencia
                     mask = np.zeros(template_image.shape[:2], dtype=np.uint8)
                     # Desenhar um círculo branco na máscara
                     cv2.circle(mask, (center_x, center_y), radius, 255, -1)
                     # Aplicar a máscara na imagem
                     hsv_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2HSV)
                     hsv_roi_ref = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
                     # Calcular a média apenas dos pixels dentro do círculo
                     mean_ref = cv2.mean(hsv_roi_ref, mask=mask)[:3]

                     # Criar uma máscara do mesmo tamanho da imagem da peça
                     mask = np.zeros(image.shape[:2], dtype=np.uint8)
                     # Desenhar um círculo branco na máscara
                     cv2.circle(mask, (center_x, center_y), radius, 255, -1)
                     # Aplicar a máscara na imagem
                     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                     hsv_roi_ref = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
                     # Calcular a média apenas dos pixels dentro do círculo
                     mean = cv2.mean(hsv_roi_ref, mask=mask)[:3]

                     difference = color_difference(mean_ref,mean)


                     # Armazenar os valores de referência
                     #reference_colors.append(mean_color_ref)
              else: 
                     roi_ref = template_image[y:y+h, x:x+w]
                     hsv_roi_ref = cv2.cvtColor(roi_ref, cv2.COLOR_BGR2HSV)
                     mean_ref = cv2.mean(hsv_roi_ref)[:3]

                     roi_ref = image[y:y+h, x:x+w]
                     hsv_roi_ref = cv2.cvtColor(roi_ref, cv2.COLOR_BGR2HSV)
                     mean = cv2.mean(hsv_roi_ref)[:3] 
                     
                     difference = color_difference(mean_ref,mean)
              
              # Se atender ao critério, marca como OK e sai do loop
              if difference < cor_tolerancia:
                     status_cor = True
                     break


                     

        except Exception as e:
         print(f"Erro ao processar {template_file}: {str(e)}")
         continue
     return status_cor