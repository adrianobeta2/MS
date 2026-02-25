import cv2
import numpy as np
import configparser

# Função para calcular a diferença entre duas cores (em HSV)
def color_difference(hsv1, hsv2):
    return np.linalg.norm(np.array(hsv1) - np.array(hsv2))

# Configurar o programa (pode ser alterado conforme necessário)
programa = 1

# Carregar a imagem de referência com base no programa
match programa:
    case 1:
        template_image = cv2.imread('ref_programa1.png') 
    case 2:
        template_image = cv2.imread('ref_programa2.png')
    case 3:
        template_image = cv2.imread('ref_programa3.png') 
    case 4:
        template_image = cv2.imread('ref_programa4.png')  
    case 5:
        template_image = cv2.imread('ref_programa5.png')
    case 6:
        template_image = cv2.imread('ref_programa6.png') 
    case 7:
        template_image = cv2.imread('ref_programa7.png') 
    case 8:
        template_image = cv2.imread('ref_programa8.png')
    case 9:
        template_image = cv2.imread('ref_programa9.png') 
    case 10:
        template_image = cv2.imread('ref_programa10.png') 
    case _:
        print("Caso não encontrado.")
        exit(1)

# Verificar se a imagem de referência foi carregada corretamente
if template_image is None:
    print("Erro: Imagem de referência não encontrada ou inválida.")
    exit(1)

# Ler o arquivo de configuração INI
config = configparser.ConfigParser()
config.read('config.ini')

# Obter as coordenadas das ROIs para a imagem de referência
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

# Calcular as cores médias das ROIs na imagem de referência
reference_colors = []
for (x, y, w, h) in rois:
    roi_ref = template_image[y:y+h, x:x+w]
    hsv_roi_ref = cv2.cvtColor(roi_ref, cv2.COLOR_BGR2HSV)
    mean_color_ref = cv2.mean(hsv_roi_ref)[:3]
    reference_colors.append(mean_color_ref)

# Carregar a imagem da peça
image = cv2.imread('captura.png')  # Substitua pelo caminho da sua imagem de teste

# Verificar se a imagem da peça foi carregada corretamente
if image is None:
    print("Erro: Imagem da peça não encontrada ou inválida.")
    exit(1)

# Definir um limiar de tolerância para a comparação
threshold = 30  # Tolerância para diferença de cor (ajuste conforme necessário)

# Analisar cada ROI na imagem da peça
for i, (x, y, w, h) in enumerate(rois):
    # Recortar a ROI da imagem
    roi = image[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean_color = cv2.mean(hsv_roi)[:3]
    
    # Comparar com a cor de referência correspondente
    difference = color_difference(reference_colors[i], mean_color)
    
    # Verificar se a cor está dentro do padrão
    if difference < threshold:
        print(f"ROI {i+1}: Cor dentro do padrão.")
        color = (0, 255, 0)  # Verde para OK
    else:
        print(f"ROI {i+1}: Cor fora do padrão.")
        color = (0, 0, 255)  # Vermelho para fora do padrão
    
    # Desenhar a ROI na imagem original
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

# Mostrar a imagem com as ROIs destacadas
cv2.imshow("Imagem com ROIs", image)

# Aguardar a tecla pressionada para fechar a janela
cv2.waitKey(0)
cv2.destroyAllWindows()

