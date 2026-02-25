import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_res(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def configurar_camera_2(img, brilho=0, contraste=1, exposicao=0.6):
    img_adjusted = cv2.convertScaleAbs(img, alpha=contraste, beta=brilho)
    img_adjusted = cv2.convertScaleAbs(img_adjusted, alpha=1, beta=exposicao*100)
    return img_adjusted

# Carregar imagem
imagem = 'ref_programa1.png'
img = cv2.imread(imagem, cv2.IMREAD_GRAYSCALE)

# Configurações da ROI (ajuste conforme necessário)
x_initial, y_initial = 327, 236
width, height = 50, 14

# Pré-processamento (ajuste esses valores)
img_adjusted = configurar_camera_2(img, brilho=20, contraste=2, exposicao=0.01)
gamma_corrected = gamma_res(img_adjusted, gamma=0.01)
roi = gamma_corrected[y_initial:y_initial+height, x_initial:x_initial+width]

# Processamento da ROI
roi_equalized = cv2.equalizeHist(roi)
roi_blurred = cv2.GaussianBlur(roi_equalized, (3, 3), 0)  # Kernel menor

# Ajuste fino dos parâmetros do Canny
edges_roi = cv2.Canny(roi_blurred, threshold1=20, threshold2=40)  # Valores mais baixos

# Visualização intermediária (debug)
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(roi, cmap='gray'), plt.title('ROI Original')
plt.subplot(132), plt.imshow(roi_equalized, cmap='gray'), plt.title('ROI Equalizada')
plt.subplot(133), plt.imshow(edges_roi, cmap='gray'), plt.title('Bordas Detectadas')
plt.show()

# Detecção de linhas com parâmetros ajustados
linesP = cv2.HoughLinesP(edges_roi, 
                        rho=1, 
                        theta=np.pi/180, 
                        threshold=10,  # Valor mais baixo
                        minLineLength=3,  # Menor comprimento
                        maxLineGap=2)    # Menor gap permitido

# Preparar imagem para desenho
roi_with_lines = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
mean_angle = 0

if linesP is not None:
    angles = []
    for line in linesP:
        x1, y1, x2, y2 = line[0]
        cv2.line(roi_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 1)
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)
    
    if angles:
        mean_angle = np.mean(angles)
        print(f"Ângulo médio detectado: {mean_angle:.2f} graus")
        print(f"Total de linhas detectadas: {len(linesP)}")
    else:
        print("Nenhuma linha detectada após filtragem")
else:
    print("Nenhuma linha detectada")

# Visualização final
img_with_lines = cv2.cvtColor(gamma_corrected, cv2.COLOR_GRAY2BGR)
img_with_lines[y_initial:y_initial+height, x_initial:x_initial+width] = roi_with_lines
cv2.rectangle(img_with_lines, (x_initial, y_initial), 
             (x_initial + width, y_initial + height), (0, 255, 0), 1)

plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
plt.title(f'Linhas Detectadas - Ângulo médio: {mean_angle:.2f}°')
plt.axis('off')
plt.show()