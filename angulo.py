import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def detecta_angulo_e_linhas(roi):
    # Aplicar equalização de histograma na ROI para melhorar o contraste
    roi_equalized = cv2.equalizeHist(roi)

    # Aplicar um filtro de suavização (GaussianBlur) para reduzir o ruído
    roi_blurred = cv2.GaussianBlur(roi_equalized, (5, 5), 0)

    # Aplicar o algoritmo de Canny para detectar bordas
    edges_roi = cv2.Canny(roi_blurred, threshold1=50, threshold2=150)

    # Aplicar a Transformada de Hough na ROI para detectar linhas
    lines = cv2.HoughLines(edges_roi, 1, np.pi/180, 100)

    # Criar uma cópia colorida da ROI para desenhar as linhas detectadas
    roi_with_lines = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

    mean_angle = 0.0  # Valor padrão caso nenhuma linha seja detectada

    # Se linhas foram detectadas, desenhá-las na ROI
    if lines is not None:
        angles = []  # Lista para armazenar os ângulos
        for rho, theta in lines[:, 0]:
            # Converter coordenadas polares para cartesianas
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            # Gerar uma cor aleatória para a linha
            color = (0, 0, 255)
            
            # Desenhar a linha na imagem colorida
            cv2.line(roi_with_lines, (x1, y1), (x2, y2), color, 2)
            
            # Calcular o ângulo da linha em graus
            angle = np.degrees(theta)
            angles.append(angle)  # Armazenar o ângulo na lista
        
        # Calcular a média dos ângulos
        mean_angle = np.mean(angles)
        print(f"Média dos ângulos detectados: {mean_angle:.2f} graus")

    return roi_with_lines, mean_angle  # Corrigido: alinhado com o início da função