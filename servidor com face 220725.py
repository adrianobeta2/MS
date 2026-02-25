from flask import Flask,abort, request, Response, jsonify, send_file,stream_with_context, render_template
import configparser
import cv2
import threading
import os
from flask_cors import CORS
import requests
import json
import face_recognition
import numpy as np
import time
from datetime import datetime
import centroPonto
import Localizador
import AnaliseCor
import angulo
import AnalisePixel
import AnaliseParafuso
import AnaliseTextura
import glob
import re
import base64
from pypylon import pylon
from werkzeug.utils import secure_filename
app = Flask(__name__)

# Variáveis globais
camera_ociosa = False
ultima_atividade = None
monitorando = True
capture_thread_running = True
PASTA_ROSTOS = 'rostos_cadastrados'

def carregar_rostos(pasta=PASTA_ROSTOS):
    rostos_conhecidos = []
    nomes = []
    if not os.path.exists(pasta):
        return rostos_conhecidos, nomes

    for arquivo in os.listdir(pasta):
        if arquivo.endswith(".npy"):
            rostos_conhecidos.append(np.load(os.path.join(pasta, arquivo)))
            nomes.append(os.path.splitext(arquivo)[0])
    return rostos_conhecidos, nomes

def check_camera_ociosa():
    global camera_ociosa, ultima_atividade, monitorando, capture_thread_running
    
    while True:
        tempo_atual = datetime.now()
        
        if ultima_atividade is not None and monitorando:
            tempo_ocioso = (tempo_atual - ultima_atividade).total_seconds()
            
            if tempo_ocioso > 900:  # 15 minutos em segundos
                print("Passou mais de 15 minutos sem atividade. Desligando a câmera.")
                camera_ociosa = True
                
                # Primeiro para a thread de captura
                capture_thread_running = False
                time.sleep(0.1)  # Dá um tempo para a thread encerrar
                
                try:
                    if(cam_tipo == "basler"):
                       camera.Close()
                    else:
                       camera.release()
                except:
                    print("Erro ao liberar câmera")
                monitorando = False  # Para o loop de monitoramento
                
        time.sleep(10)  # Verifica a cada 10 segundos
           


# Criando e iniciando a thread(nao usado temporariamente)
thread_cam_ociosa = threading.Thread(target=check_camera_ociosa)
thread_cam_ociosa.daemon = True  # Encerra a thread quando o programa principal terminar
thread_cam_ociosa.start() 

def get_serial():
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("Serial"):
                    return line.strip().split(":")[1].strip()
    except Exception:
        return None

serial_autorizado = None #"163d90eebde2841a" #"" # Defina o serial autorizado "100000003671bd44"

@app.before_request
def verificar_serial():
    serial_atual = get_serial()
    if serial_atual != serial_autorizado:
        abort(403, description="Acesso negado! Dispositivo não autorizado")

#verificar_serial()
CORS(app)

def salvar_imagem_com_log(image, base_dir='log_imagens'):
    # Obtém data e hora atual
    agora = datetime.now()
    ano = agora.strftime("%Y")
    mes = agora.strftime("%m")
    dia = agora.strftime("%d")
    hora_nome = agora.strftime("%H-%M-%S")  # nome da imagem

    # Cria o caminho completo com base em ano/mes/dia
    caminho_pasta = os.path.join(base_dir, ano, mes, dia)
    os.makedirs(caminho_pasta, exist_ok=True)  # Cria diretórios se não existirem

    # Caminho completo da imagem
    caminho_imagem = os.path.join(caminho_pasta, f"{hora_nome}.jpg")

    # Salva a imagem
    cv2.imwrite(caminho_imagem, image)
    print(f"Imagem salva em: {caminho_imagem}")

import os

import os
from datetime import datetime
import cv2

import os
from datetime import datetime
import cv2

def salvar_imagem_no_pendrive(image, pendrive_path='/media/usb'):
    print(f"[DEBUG] Verificando ponto de montagem: {pendrive_path}")
    print(f"[DEBUG] Resultado ismount: {os.path.ismount(pendrive_path)}")
    
    if not os.path.ismount(pendrive_path):
        print("Erro: Pendrive não montado.")
        return

    agora = datetime.now()
    ano = agora.strftime("%Y")
    mes = agora.strftime("%m")
    dia = agora.strftime("%d")
    hora_nome = agora.strftime("%H-%M-%S")

    caminho_pasta = os.path.join(pendrive_path, "log_imagens", ano, mes, dia)
    print(f"[DEBUG] Caminho da pasta final: {caminho_pasta}")

    try:
        os.makedirs(caminho_pasta, exist_ok=True)
    except Exception as e:
        print(f"[ERRO] ao criar pasta: {e}")
        return

    caminho_imagem = os.path.join(caminho_pasta, f"{hora_nome}.jpg")
    print(f"[DEBUG] Caminho da imagem: {caminho_imagem}")

    try:
        if cv2.imwrite(caminho_imagem, image):
            print(f"Imagem salva com sucesso em: {caminho_imagem}")
        else:
            print("Erro ao salvar a imagem com cv2.imwrite.")
    except Exception as e:
        print(f"[ERRO] ao salvar imagem: {e}")



def get_serial():
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("Serial"):
                    return line.strip().split(":")[1].strip()
    except Exception:
        return None

# Defina o Serial Number da câmera desejada
# serial_number = "22900012"  # Altere para o número da sua câmera

@app.before_request
def verificar_serial():
    serial_atual = get_serial()
    if serial_atual != serial_autorizado:
        abort(403, description="Acesso negado! Dispositivo não autorizado.")

config = configparser.ConfigParser()
config.read("cameras.ini")

# Obter os caminhos das câmeras
cam_path = config.get("CAMERAS", "cam") #comando para listar as cameras disponiveis:  ls -l /dev/v4l/by-id/
cam_tipo = config.get("CAMERAS", "tipo")
serial_number = config.get("CAMERAS","serial_number_basler")
# Variável global para a câmera
camera = None
lock = threading.Lock()  # Para evitar concorrência nos frames
last_frame = None


# Configuração de tamanho da imagem
TARGET_WIDTH = 640  # Largura desejada
TARGET_HEIGHT = 480  # Altura desejada
JPEG_QUALITY = 80  # Qualidade do JPEG (0 a 100, menor = mais rápido)


CONFIG_FILE_CAM = "cameras.ini"

def save_camera_config(camera_type, serial_number=None):
    """ Atualiza a configuração da câmera sem apagar outras chaves """
    config = configparser.ConfigParser()

    # Se o arquivo já existir, lê antes para manter outras chaves
    if os.path.exists(CONFIG_FILE_CAM):
        config.read(CONFIG_FILE_CAM)

    if "CAMERAS" not in config:
        config["CAMERAS"] = {}

    config["CAMERAS"]["tipo"] = camera_type

    if camera_type.lower() == "basler" and serial_number:
        config["CAMERAS"]["serial_number_basler"] = serial_number
    

    with open(CONFIG_FILE_CAM, "w") as configfile:
        config.write(configfile)

def get_available_basler_cameras():
    """ Retorna os números de série das câmeras Basler conectadas """
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    return [device.GetSerialNumber() for device in devices]

def initialize_camera():
    global camera,capture_thread_running
    camera = cv2.VideoCapture(cam_path)
    #camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 15)
    capture_thread_running = True
    threading.Thread(target=camera_loop, daemon=True).start()

def initialize_camera_basler():
    """ Inicializa a câmera Basler e inicia a thread de captura """
    global camera, capture_thread_running
    

    
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    
    # Verifica se a câmera com o Serial Number está disponível
    for device in devices:
        if device.GetSerialNumber() == serial_number:
            camera = pylon.InstantCamera(tl_factory.CreateDevice(device))
            break

    if camera is None:
        print("Erro: Câmera não encontrada.")
        return
    
    camera.Open()  # <-- Importante abrir a câmera para configurar parâmetros
    camera.MaxNumBuffer.SetValue(3)
    camera.OutputQueueSize.SetValue(1)
    

    camera.AcquisitionMode.Value = "Continuous"  # Captura contínua
    camera.AcquisitionFrameRateEnable.Value = True
    camera.AcquisitionFrameRate.Value = 16.0
  
    
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
   

    
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    capture_thread_running = True
    threading.Thread(target=camera_loop_basler, args=(converter,), daemon=True).start()

# Função para calcular a diferença entre duas cores (em HSV)
def color_difference(hsv1, hsv2):
    return np.linalg.norm(np.array(hsv1) - np.array(hsv2))


def camera_loop():
    global camera, last_frame,capture_thread_running
    while True and capture_thread_running:
        with lock:
            if camera and camera.isOpened():
                ret, frame = camera.read()
                if ret:
                    last_frame = frame.copy()
        time.sleep(0.03)  # Reduz a carga na CPU



def camera_loop_basler(converter):
    """ Captura frames continuamente e redimensiona antes de armazenar """
    global last_frame, capture_thread_running
    
    while camera.IsGrabbing() and capture_thread_running:
            try:
                grabResult = camera.RetrieveResult(500, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    image = converter.Convert(grabResult)
                    frame = image.GetArray()

                    # Redimensionar a imagem para um tamanho menor
                    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

                    with lock:
                        last_frame = frame
                grabResult.Release()
                time.sleep(0.01)
            except Exception as e:
                print(f"[ERRO] Falha na captura da câmera Basler: {e}")
                print("[INFO] Tentando reconectar a câmera em 3 segundos...")

                try:
                    if camera.IsGrabbing():
                        camera.StopGrabbing()
                    camera.Close()
                except Exception as e_close:
                    print(f"[WARN] Erro ao fechar câmera: {e_close}")

                time.sleep(3)
                initialize_camera_basler()
                break  # Sai da thread atual; a nova será iniciada pela função acima



def adjust_image_optimized(img, contrast=1.0, brightness=0, gamma=1.0):
    """
    Ajusta contraste, brilho e gama de uma imagem em um único passo.
    """
    # Converter para escala de cinza e ajustar contraste/brilho
    adjusted = cv2.convertScaleAbs(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), alpha=contrast, beta=brightness)

    # Aplicar equalização de histograma para realçar contraste
    # adjusted = cv2.equalizeHist(adjusted)

   
    blurred = cv2.GaussianBlur(adjusted, (5, 5), 0)
    adjusted = cv2.addWeighted(adjusted, 1.5, blurred, -0.5, 0)
    
    # Aplicar correção de gama (tabela de lookup otimizada)
    gamma_correction = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype="uint8")
    return cv2.LUT(adjusted, gamma_correction)



import cv2
import numpy as np

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


# Função para desenhar a ROI (retângulo ou círculo)
def desenhar_roi(imagem, x, y, w, h, status, posicao, forma="retangulo"):
    """
    Desenha uma ROI na imagem com base no status e na forma (retângulo ou círculo).
    
    :param imagem: Imagem onde a ROI será desenhada.
    :param x, y: Coordenadas do canto superior esquerdo da ROI.
    :param w, h: Largura e altura da ROI (para retângulo) ou raio (para círculo).
    :param status: Status da ROI (True ou False).
    :param cor: Se True, desenha na imagem colorida; se False, desenha na imagem em escala de cinza.
    :param forma: Forma da ROI ("retangulo" ou "circulo").
    """
    if(posicao):
        cor_roi = (255, 0, 0)
    else:
       # Definir a cor com base no status
       cor_roi = (0, 255, 0) if status else (0, 0, 255)  # Verde para True, Vermelho para False

    # Desenhar a ROI com base na forma
    if forma == "retangulo":
        cv2.rectangle(imagem, (x, y), (x + w, y + h), cor_roi, 1)
    elif forma == "circulo":
        centro = (x , y )  # Centro do círculo
        raio = w              # Raio do círculo
        cv2.circle(imagem, centro, raio, cor_roi, 1)
    else:
        raise ValueError("Forma não suportada. Use 'retangulo' ou 'circulo'.")



url = "http://127.0.0.1:6001/capture"

@app.route('/')
def home():
    return render_template('index.html')

# Rota para a página limites.html
@app.route('/limites')
def limites():
    return render_template('limites.html')

# Rota para a página configuracoes.html
@app.route('/configuracoes')
def configuracoes():
    return render_template('configuracoes.html')


# Rota para a página limites.html
@app.route('/localizador')
def localizador():
    return render_template('localizador.html')

@app.route('/executar', methods=['POST'])
def processResult():
    try:
        global monitorando
        global camera_ociosa
        global ultima_atividade
        monitorando = True
        
        ultima_atividade = datetime.now() 
        # Verifica se a câmera está ociosa e reinicializa se necessário
        if (camera_ociosa):
            camera_ociosa = False
            if(cam_tipo == "basler"):
                initialize_camera_basler()
            else:
                initialize_camera()
           
        
        tick_start = cv2.getTickCount()
        # Obtém a data e hora atuais
        
        # Processar a imagem como no código original
        response = requests.get(url)
        #response = requests.capture_frame()
        if response.status_code == 200:
            nparr = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            return jsonify({"error": f"Erro ao capturar frame: {response.json()['error']}"})
       
        #programa = request.json.get('programa')  # Se estiver usando JSON no corpo
        data = request.json  # Receber JSON do corpo
        programa = data.get('programa')  # Acessar o valor do programa
        # Ler o arquivo de configuração INI
        config = configparser.ConfigParser()
        config.read(f'config_{programa}.ini')
        
        # Obter os parâmetros de ajuste de imagem
        contrast = float(config['ImageAdjustments']['contrast'])
        brightness = int(config['ImageAdjustments']['brightness'])
        gamma = float(config['ImageAdjustments']['gamma'])
        exposure = float(config['ImageAdjustments']['exposure'])

        # Obter as coordenadas da referência para o ponto do parafuso
        x_ref = int(config['Referencia']['x_ref']) 
        y_ref = int(config['Referencia']['y_ref'])
        width_ref = int(config['Referencia']['width_ref'])
        height_ref = int(config['Referencia']['height_ref'])
        x_interes = int(config['Referencia']['x_interes'])
        y_interes = int(config['Referencia']['y_interes'])
        width_interes = int(config['Referencia']['width_interes'])
        height_interes = int(config['Referencia']['height_interes'])

        roi_regiao_template = (x_interes, y_interes, width_interes, height_interes)

        ref_posicao_templ = (x_ref, y_ref) #fator para ajuste de posicao
       
        # Obter valores booleanos
        cor = config.getboolean('Ferramentas', 'cor')
        textura = config.getboolean('Ferramentas', 'textura')
        pixel = config.getboolean('Ferramentas', 'pixel')
        ponto_parafuso = config.getboolean('Ferramentas', 'ponto_parafuso')
        ponto_xy = config.getboolean('Ferramentas', 'ponto_xy')
        posicao = config.getboolean('Ferramentas', 'posicao')
        n_rois = config.getint('Ferramentas', 'n_rois')
        nome_programa =  config.get('Ferramentas', 'nome_programa')


        if(posicao):
            template = cv2.imread(f'template_{programa}.png', 0)

        # Carregar a imagem toral de referência com base no programa
        if ponto_parafuso == True or textura  == True or posicao == True or cor == True or ponto_xy == True or pixel == True:
                template_image = None
                template_image_NOK = None
                match programa:
                        case 1:
                            template_image = cv2.imread('ref_programa1_OK.png') 
                            template_image_NOK = cv2.imread('ref_programa1_NOK.png')
                        case 2:
                            template_image = cv2.imread('ref_programa2_OK.png')
                            template_image_NOK = cv2.imread('ref_programa2_NOK.png')
                        case 3:
                            template_image = cv2.imread('ref_programa3_OK.png') 
                            template_image_NOK = cv2.imread('ref_programa3_NOK.png')
                        case 4:
                            template_image = cv2.imread('ref_programa4_OK.png')  
                            template_image_NOK = cv2.imread('ref_programa4_NOK.png')
                        case 5:
                            template_image = cv2.imread('ref_programa5_OK.png')
                            template_image_NOK = cv2.imread('ref_programa5_NOK.png')
                        case 6:
                            template_image = cv2.imread('ref_programa6_OK.png') 
                            template_image_NOK = cv2.imread('ref_programa6_NOK.png')
                        case 7:
                            template_image = cv2.imread('ref_programa7_OK.png') 
                            template_image_NOK = cv2.imread('ref_programa7_NOK.png')
                        case 8:
                            template_image = cv2.imread('ref_programa8_OK.png')
                            template_image_NOK = cv2.imread('ref_programa8_NOK.png')
                        case 9:
                            template_image = cv2.imread('ref_programa9_OK.png') 
                            template_image_NOK = cv2.imread('ref_programa9_NOK.png')
                        case 10:
                            template_image = cv2.imread('ref_programa10_OK.png') 
                            template_image_NOK = cv2.imread('ref_programa10_NOK.png')
                        case _:
                            return jsonify({"error": "Programa não encontrado."}), 404
                    # Verificar se a imagem de referência foi carregada corretamente
                if template_image_NOK is None:
                    return jsonify({"error": "Imagem de referência não encontrada."}), 400
                
               
        # Carregar a imagem da peça
        image = img  # Imagem para o teste de cor e parafuso 

        # Obter as coordenadas das ROIs
        rois = []
        for i in range(1, n_rois+1):  # Três ROIs
            section_name = f'ROI{i}'
            if section_name in config:
                x = int(config[section_name]['x'])
                if(x == None):
                     x = int(config[section_name]['x_anterior'])

                y = int(config[section_name]['y'])
                if(y == None):
                     y = int(config[section_name]['y_anterior'])

                w = int(config[section_name]['width'])
                if(w == None):
                     w = int(config[section_name]['width_anterior'])
                     
                h = int(config[section_name]['height'])
                if(h == None):
                     h = int(config[section_name]['height_anterior'])
                textura_tolerancia  = int(config[section_name]['textura_tolerancia'])
                pixel_tolerancia = int(config[section_name]['pixel_tolerancia'])          
                threshold_cor = int(config[section_name]['threshold_cor'])
                ponto_tolerancia = int(config[section_name]['ponto_tolerancia'])
                
                roi_incial = (x, y, w, h)
                
                (x_inicial,y_inicial) = (x, y)

                if(posicao): # obs.: Posição = Localizador
                   roi_final, roi_template = Localizador.ajustePosicao(template,roi_regiao_template,img,ref_posicao_templ,roi_incial)
                   x, y, w, h = roi_final
                   (x_final,y_final)  = (x, y)

                   
                   delta_posicao_x = np.abs(x_final - x_inicial)
                   delta_posicao_y = np.abs(y_final - y_inicial)
                   if(delta_posicao_x or delta_posicao_y) >20: # um fator de 20 pixels.
                       status_posicao = False
                   else:
                       status_posicao = True
                else:
                    status_posicao = True

                rois.append((x, y, w, h,textura_tolerancia,pixel_tolerancia,threshold_cor,ponto_tolerancia))
        
            

        # Carregar a imagem referencia
        image_ref = template_image  # Imagem para o teste de cor e parafuso 
        
           # Carregar a imagem referencia NOK
        image_ref_nok = template_image_NOK

        # ja inicia arquivo de status
        config = configparser.ConfigParser()


        status_list = []  # Lista para armazenar os valores de status  

        # Carregar o arquivo status.ini se ele existir
        config.read('status.ini')

        # Verificar se a seção "STATUS" já existe, caso contrário, criá-la
        if 'STATUS' not in config:
            config['STATUS'] = {}


        rois_data = []
        for i, (x, y, w, h,textura_tolerancia,pixel_tolerancia, threshold_cor,ponto_tolerancia) in enumerate(rois, start=1):
            circulo = False
            if(w==h):circulo = True
            
            #########################Análise de Cor ######################################
            if(cor == True):
                coordenadas = x,y,w,h
              
                #analise imagem atual
                status_cor = AnaliseCor.Cor(image,programa,threshold_cor,coordenadas)
            else:
                status_cor = True
            ##############################################################################

            # Ajustar a imagem para pixel e textura
                   
            gray = adjust_image_optimized_2(img, contrast, brightness, gamma, exposure)
            
            if(circulo):
                # Criar uma máscara preta do tamanho da imagem
                mask = np.zeros_like(gray, dtype=np.uint8)
                # Desenhar um círculo branco na máscara
                cv2.circle(mask, (x, y), w, 255, -1)
                roi = cv2.bitwise_and(gray, gray, mask=mask)
            else:
                mask = None
                roi = gray[y:y+h, x:x+w]
                # Converter imagem em escala de cinza para BGR
            
            status = False
           
            ################## Análise de textura #######################################
            if textura == True:
                coordenadas = x,y,w,h
                status_texutura,media_textura = AnaliseTextura.textura_analise_OK(roi,programa,gray,mask,coordenadas, textura_tolerancia, contrast, brightness, gamma, exposure)
                if(status_texutura==False):
                   status_texutura, media_textura = AnalisePixel.pixel_analise_NOK(roi,programa,gray,mask,coordenadas,textura_tolerancia,contrast, brightness, gamma, exposure)
                   if(status_texutura==True):
                       status_texutura = False 
            else:
                status_texutura = True
            
            ######################## Análise de pixel (Quantify)###############################

            coordenadas = x,y,w,h
            if pixel == True:
               
               status_roi,media_roi = AnalisePixel.pixel_analise_OK(roi,programa,gray,mask,coordenadas, pixel_tolerancia, contrast, brightness, gamma, exposure)

               if(status_roi==False):
                   status_roi, media_roi = AnalisePixel.pixel_analise_NOK(roi,programa,gray,mask,coordenadas,pixel_tolerancia,contrast, brightness, gamma, exposure)
                   if(status_roi==True):
                       status_roi = False                     
            else:
              status_roi = True

            cX = 0
            cY = 0
            ################## Análise de ponto do parafuso - retorna a coordenadas x e y #######################################
            if ponto_parafuso == True:
                
                cX,cY,status_paratuso = AnaliseParafuso.ParafusoAnaliseOK(image,programa,coordenadas,contrast,brightness,gamma,exposure,ponto_tolerancia)

                if(status_paratuso == False):
                    cX,cY,status_paratuso = AnaliseParafuso.ParafusoAnaliseNOK(image,programa,coordenadas,contrast,brightness,gamma,exposure,ponto_tolerancia)
                    if(status_paratuso == True):
                        status_paratuso = False
                        

            else:
                status_paratuso = True
            
             ##################   Analise de ponto - retorna as coordenadas x e y #######################################
            if ponto_xy == True:

                #analise da imagem de referencia OK
                status_ponto_xy = False
                img_regiao_paraf,  cX, cY, min_val = centroPonto.centroDoponto(image,(x,y),w,contrast,brightness,gamma,exposure,ponto_tolerancia)
                if(cX != 0 and cY !=0):
                    status_ponto_xy = True
                       
            else:
                status_ponto_xy = True



            if( (status_texutura and status_roi and status_cor and status_paratuso and status_posicao and status_ponto_xy) ):
                status = True
            
            if(textura):
               media_textura = f"{media_textura:.2f}"
            else:
               media_textura = f"{0:.2f}"

            if(pixel):
               media_roi = f"{media_roi:.2f}"
            else:
               media_roi = f"{0:.2f}"
            
            if(circulo):
                forma = "circulo"
            else:
                forma = "retangulo"
            
            
            
            # Atualizar os valores da seção para enviar para a tela de status.
            config['STATUS'][f'parcial_p{i}'] = str(status)   

            desenhar_roi(image, x, y, w, h, status, False, forma)   

            status_list.append(status)  # Armazena o status para verificar depois
            
            if posicao:
                x_t, y_t = roi_template
                desenhar_roi(image, x_t, y_t, width_ref, height_ref, status, posicao, "retangulo")
            rois_data.append({
                "roi_index": i,
                "media_textura": media_textura,
                "textura_status": status_texutura,
                "media_roi": media_roi,
                "pixel_status": status_roi,
                "cor_status":status_cor,
                "status":status,
                "textura_ativo":textura,
                "cor_ativo":cor,
                "pixel_ativo":pixel,
                "forma":forma,
                "parafuso_ativo":ponto_parafuso,
                "parafuso":status_paratuso,
                "posicao_ativo": posicao,
                "posicao": status_posicao,
                "ponto_xy_ativo": ponto_xy,
                "ponto_xy": status_ponto_xy,
                "ponto_x":cX,
                "ponto_y":cY
            })
        
        if(cor):
            cv2.imwrite('imagem_status.jpg', image)
        else:
            cv2.imwrite('imagem_status.jpg', image)
        
        salvar_imagem_no_pendrive(image)
        resultado_geral = all(status_list)  # Verifica se todos os status são True
        
        tick_end = cv2.getTickCount()
        time_taken = (tick_end - tick_start) / cv2.getTickFrequency() * 1000  # Convertendo para ms
        time_taken = round(time_taken, 2)  # Arredondar para 2 casas decimais
        print(f"Tempo de execução: {time_taken} ms")
        
        # Finaliza pra indicar que o programa foi executado
        config['STATUS']['executado'] = "true"
        # Finaliza pra indicar que o programa foi executado
        config['STATUS']['n_rois'] = str(n_rois)
        config['STATUS']['status_geral'] = str(resultado_geral)
        config['STATUS']['tempo_execucao'] = str(time_taken)
        config['STATUS']['programa'] = str(programa)
        config['STATUS']['nome_programa'] = nome_programa
        
        # Salvar as alterações no arquivo INI
        with open('status.ini', 'w') as configfile:
              config.write(configfile)
       
    
       
        
        return jsonify({
            "rois_data": rois_data
            
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_references_ok', methods=['POST'])
def delete_reference_files_ok():
    data = request.json
    programa = data.get('programa')
    
    if not programa:
        return jsonify({"error": "Número do programa não fornecido"}), 400
    
    base_filename = f"ref_programa{programa}_OK"
    deleted_files = []
    errors = []
    
    # Padrão para encontrar todos os arquivos relacionados ao programa
    pattern = f"{base_filename}*.png"
    
    # Encontra e deleta todos os arquivos que correspondem ao padrão
    for filename in glob.glob(pattern):
        try:
            os.remove(filename)
            deleted_files.append(filename)
        except Exception as e:
            errors.append(f"Erro ao deletar {filename}: {str(e)}")
    
    if deleted_files:
        message = f"Arquivos deletados: {', '.join(deleted_files)}"
        if errors:
            message += f" | Erros: {'; '.join(errors)}"
        return jsonify({"message": message}), 200
    elif errors:
        return jsonify({"error": "; ".join(errors)}), 500
    else:
        return jsonify({"message": f"Nenhum arquivo encontrado para o programa {programa}"}), 200
    

@app.route('/delete_references_nok', methods=['POST'])
def delete_reference_files_nok():
    data = request.json
    programa = data.get('programa')
    
    if not programa:
        return jsonify({"error": "Número do programa não fornecido"}), 400
    
    base_filename = f"ref_programa{programa}_NOK"
    deleted_files = []
    errors = []
    
    # Padrão para encontrar todos os arquivos relacionados ao programa
    pattern = f"{base_filename}*.png"
    
    # Encontra e deleta todos os arquivos que correspondem ao padrão
    for filename in glob.glob(pattern):
        try:
            os.remove(filename)
            deleted_files.append(filename)
        except Exception as e:
            errors.append(f"Erro ao deletar {filename}: {str(e)}")
    
    if deleted_files:
        message = f"Arquivos deletados: {', '.join(deleted_files)}"
        if errors:
            message += f" | Erros: {'; '.join(errors)}"
        return jsonify({"message": message}), 200
    elif errors:
        return jsonify({"error": "; ".join(errors)}), 500
    else:
        return jsonify({"message": f"Nenhum arquivo encontrado para o programa {programa}"}), 200
    

@app.route('/processWithImage', methods=['GET'])
def process_image():
    try:
        
        # exemplo de endpoint http://localhost:6001/processWithImage?programa=1.

        programa = request.args.get('programa')  # Usando query string para GET
        # Ler o arquivo de configuração INI
        config = configparser.ConfigParser()
        config.read(f'config_{programa}.ini')
        # Obter os parâmetros de ajuste de imagem
        contrast = float(config['ImageAdjustments']['contrast'])
        brightness = int(config['ImageAdjustments']['brightness'])
        gamma = float(config['ImageAdjustments']['gamma'])
        exposure = float(config['ImageAdjustments']['exposure'])

        # Obter as coordenadas das ROIs
        rois = []
        for i in range(1, 4):  # Três ROIs
            section_name = f'ROI{i}'
            if section_name in config:
                x = int(config[section_name]['x'])
                if(x == None):
                     x = int(config[section_name]['x_anterior'])

                y = int(config[section_name]['y'])
                if(y == None):
                     y = int(config[section_name]['y_anterior'])

                w = int(config[section_name]['width'])
                if(w == None):
                     w = int(config[section_name]['width_anterior'])

                h = int(config[section_name]['height'])
                if(h == None):
                     h = int(config[section_name]['height_anterior'])
                textura_tolerancia  = int(config[section_name]['textura_tolerancia'])
                pixel_tolerancia = int(config[section_name]['pixel_tolerancia'])
                rois.append((x, y, w, h,textura_tolerancia,pixel_tolerancia))
        # Processar a imagem como no código original
        response = requests.get(url)
        if response.status_code == 200:
            nparr = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            return jsonify({"error": f"Erro ao capturar frame: {response.json()['error']}"})

        # Ajustar a imagem
        gray = adjust_image_optimized_2(img, contrast, brightness, gamma, exposure)

        # Converter imagem em escala de cinza para BGR
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        rois_data = []
       
        # Codificar a imagem em formato JPEG
        _, img_encoded = cv2.imencode('.jpg', gray_bgr)
        img_bytes = img_encoded.tobytes()

        # Retornar a imagem como resposta binária com metadados no cabeçalho
        return Response(img_bytes, mimetype='image/jpeg', headers={
            'Content-Disposition': 'inline; filename="processed_image.jpg"',
            'rois-data': json.dumps(rois_data)  # Codificar JSON como string para o cabeçalho
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Caminhos para o arquivo INI e a imagem
CONFIG_PATH = 'config.ini'
IMAGE_PATH = 'captura.png' # mao usado.

# Função para carregar o arquivo de configuração
def load_config(PATH):
    config = configparser.ConfigParser()
    config.read(PATH)
    return config

# Função para salvar alterações no arquivo de configuração
def save_config(PATH, config):
    with open(PATH, 'w') as configfile:
        config.write(configfile)


@app.route('/capture', methods=['GET'])
def capture_frame():
    global last_frame
    with lock:
        if last_frame is not None:
            _, buffer = cv2.imencode('.jpg', last_frame)
            response = Response(buffer.tobytes(), content_type='image/jpeg')
            return response
        else:
            return jsonify({"error": "Nenhum frame disponível"}), 500


@app.route('/capture_reference', methods=['POST'])
def capture_frame_reference():
    data = request.json
    programa = data.get('programa')
    global last_frame
    
    with lock:
        if last_frame is not None:
            base_filename = f"ref_programa{programa}_OK"
            extension = ".png"
            filename = base_filename + extension
            counter = 1
            
            # Verifica se o arquivo já existe e encontra o próximo número disponível
            while os.path.exists(filename):
                filename = f"{base_filename}_{counter}{extension}"
                counter += 1
            
            # Salva o frame com o nome apropriado
            try:
                cv2.imwrite(filename, last_frame)
                return jsonify({"message": f"Frame capturado e salvo como '{filename}'"}), 200
            except Exception as e:
                return jsonify({"error": f"Erro ao salvar o frame: {str(e)}"}), 500
        else:
            return jsonify({"error": "Nenhum frame disponível"}), 500

@app.route('/capture_reference_NOK', methods=['POST'])
def capture_frame_reference_NOK():
    data = request.json
    programa = data.get('programa')
    global last_frame
    
    with lock:
        if last_frame is not None:
            base_filename = f"ref_programa{programa}_NOK"
            extension = ".png"
            filename = base_filename + extension
            counter = 1
            
            # Verifica se o arquivo já existe e encontra o próximo número disponível
            while os.path.exists(filename):
                filename = f"{base_filename}_{counter}{extension}"
                counter += 1
            
            # Salva o frame com o nome apropriado
            try:
                cv2.imwrite(filename, last_frame)
                return jsonify({"message": f"Frame capturado e salvo como '{filename}'"}), 200
            except Exception as e:
                return jsonify({"error": f"Erro ao salvar o frame: {str(e)}"}), 500
        else:
            return jsonify({"error": "Nenhum frame disponível"}), 500


@app.route('/shutdown', methods=['POST'])
def shutdown():
    """
    Libera a câmera ao desligar o servidor.
    """
    global camera
    if camera:
        with lock:
            camera.release()
    return jsonify({"message": "Câmera desligada"}), 200


def get_config_float(config, section, option, fallback=0.0):
    if not config.has_section(section):
        return fallback
    if not config.has_option(section, option):
        return fallback
    value = config.get(section, option)
    if not value.strip():  # Se estiver vazio ou só espaços
        return fallback
    try:
        return float(value)
    except ValueError:
        return fallback

def get_config_int(config, section, option, fallback=0):
    if not config.has_section(section):
        return fallback
    if not config.has_option(section, option):
        return fallback
    value = config.get(section, option)
    if not value.strip():  # Se estiver vazio ou só espaços
        return fallback
    try:
        return int(value)
    except ValueError:
        return fallback

# Endpoint GET para obter os parâmetros de configuração
@app.route('/config', methods=['GET'])
def get_config():
    
    #exemple de endpoint http://localhost:6001/config?programa=1&section=ROI1.
    programa = request.args.get('programa')  # Usando query string para GET
    # Obtém o parâmetro de consulta "section" da URL
    section = request.args.get('section')

    PATH = f'config_{programa}.ini'

    # Carrega o arquivo de configuração
    config = load_config(PATH)
    
    
    
    # Verifica se uma seção específica foi solicitada
    if section:
        if section and config.has_section(section):
            # Retorna os dados da seção solicitada
            response = {
                "contrast": get_config_float(config,'ImageAdjustments', 'contrast',fallback= 1.0),
                "brightness": get_config_int(config,'ImageAdjustments', 'brightness',fallback= 20),
                "gamma": get_config_float(config,'ImageAdjustments', 'gamma', fallback=0.5),
                "exposure": get_config_float(config,'ImageAdjustments', 'exposure',fallback=1),
                "roi": {
                "x": get_config_int(config,section, 'x') or get_config_int(config, section, 'x_anterior', fallback=0),
                "y": get_config_int(config,section, 'y') or get_config_int(config, section, 'y_anterior', fallback=0),
                "width": get_config_int(config,section, 'width') or get_config_int(config, section, 'width_anterior', fallback=0),
                "height": get_config_int(config,section, 'height') or get_config_int(config,section, 'height_anterior', fallback=0),
              }
            }
        else:
            # Retorna erro caso a seção não exista
            return jsonify({"error": f"Seção '{section}' não encontrada"}), 404
    else:
        # Retorna todas as ROIs e ajustes de imagem se nenhuma seção for especificada
        rois = {}
        for sec in config.sections():
            if sec.startswith('ROI'):
                rois[sec] = {
                    "x": config.getint(sec, 'x'),
                    "y": config.getint(sec, 'y'),
                    "width": config.getint(sec, 'width'),
                    "height": config.getint(sec, 'height'),
                }
        response = {
            "contrast": config.getfloat('ImageAdjustments', 'contrast'),
            "brightness": config.getint('ImageAdjustments', 'brightness'),
            "gamma": config.getfloat('ImageAdjustments', 'gamma'),
            "exposure": config.getfloat('ImageAdjustments', 'exposure'),
            "rois": rois,
        }
    
    return jsonify(response)

# Endpoint POST para alterar os parâmetros de configuração
@app.route('/config', methods=['POST'])
def update_config():
    
    
    
    data = request.json
    
    programa = data.get('programa')
    PATH = f'config_{programa}.ini'

    config = load_config(PATH)
    
    # Atualiza os valores gerais se fornecidos
    if 'contrast' in data:
        config['ImageAdjustments']['contrast'] = str(data['contrast'])
    if 'brightness' in data:
        config['ImageAdjustments']['brightness'] = str(data['brightness'])
    if 'gamma' in data:
        config['ImageAdjustments']['gamma'] = str(data['gamma'])
    if 'exposure' in data:
        config['ImageAdjustments']['exposure'] = str(data['exposure'])
    
    # Atualiza os valores da ROI específica se fornecidos
    if 'roi' in data and 'section' in data['roi']:
        section = data['roi']['section']  # Exemplo: "ROI1", "ROI2"
        if section in config:
            roi = data['roi']
            if 'x' in roi:
                config[section]['x'] = str(roi['x'])
                if str(roi['x']) != None:
                  config[section]['x_anterior'] = str(roi['x'])
                if str(roi['x']) == None and str(roi['x_anterior']) == None :
                  config[section]['x_anterior'] = str(10)
                  config[section]['x'] = str(10)

            if 'y' in roi:
                config[section]['y'] = str(roi['y'])
                if str(roi['y']) != None:
                  config[section]['y_anterior'] = str(roi['y'])
                if str(roi['y']) == None and str(roi['y_anterior']) == None :
                  config[section]['y_anterior'] = str(10)
                  config[section]['y'] = str(10)

            if 'width' in roi:
                config[section]['width'] = str(roi['width'])
                if str(roi['width']) != None:
                  config[section]['width_anterior'] = str(roi['width'])
                if str(roi['width']) == None and str(roi['width_anterior']) == None :
                  config[section]['width_anterior'] = str(0)
                  config[section]['width'] = str(0)

            if 'height' in roi:
                config[section]['height'] = str(roi['height'])
                if str(roi['height']) != None:
                  config[section]['height_anterior'] = str(roi['height'])
                if str(roi['height']) == None and str(roi['height_anterior']) == None :
                  config[section]['height_anterior'] = str(0)
                  config[section]['height'] = str(0)
        else:
            return jsonify({"error": f"Seção {section} não encontrada"}), 400
    
    # Salva as alterações no arquivo INI
    save_config(PATH,config)
    return jsonify({"message": "Configuração atualizada com sucesso!"})

''
# Endpoint POST para alterar os parâmetros de configuração
@app.route('/ferramentas', methods=['POST'])
def update_ferramenta_config():
    

    data = request.json

    programa = data.get('programa')  # Pegando o valor de 'programa' do JSON

    PATH = f'config_{programa}.ini'
    
    config = load_config(PATH)
    # Atualiza os valores gerais se fornecidos, garantindo que sejam armazenados como "true"/"false"
    if 'nome_programa' in data:
        config['Ferramentas']['nome_programa'] = str(data['nome_programa']).lower()
    if 'cor' in data:
        config['Ferramentas']['cor'] = str(data['cor']).lower()
    if 'textura' in data:
        config['Ferramentas']['textura'] = str(data['textura']).lower()
    if 'pixel' in data:
        config['Ferramentas']['pixel'] = str(data['pixel']).lower()
    if 'ponto_parafuso' in data:
        config['Ferramentas']['ponto_parafuso'] = str(data['ponto_parafuso']).lower()
    if 'ponto_xy' in data:
        config['Ferramentas']['ponto_xy'] = str(data['ponto_xy']).lower()
    if 'posicao' in data:
        config['Ferramentas']['posicao'] = str(data['posicao']).lower()
    if 'n_rois' in data:
        config['Ferramentas']['n_rois'] = str(data['n_rois']).lower()

    # Salva as alterações no arquivo INI
    save_config(PATH,config)
    return jsonify({"message": "Configuração atualizada com sucesso!"})

# Endpoint POST para acessar os parâmetros de configuração
@app.route('/ferramentas', methods=['GET'])
def get_ferramenta_config():
    
    #exemple de endpoint http://localhost:6001/ferramentas?programa=1.
    programa = request.args.get('programa')  # Usando query string para GET

    PATH = f'config_{programa}.ini'
    # Carrega as configurações atuais
    config = load_config(PATH)

    # Retorna os valores de cada configuração
    return jsonify({
        "nome_programa": config['Ferramentas'].get('nome_programa'), #retorna o nome do programa
        "cor": config['Ferramentas'].get('cor', 'false') == 'true',  # Converte de volta para booleano
        "textura": config['Ferramentas'].get('textura', 'false') == 'true',
        "pixel": config['Ferramentas'].get('pixel', 'false') == 'true',
        "ponto_parafuso": config['Ferramentas'].get('ponto_parafuso', 'false') == 'true',
        "ponto_xy": config['Ferramentas'].get('ponto_xy', 'false') == 'true',
        "posicao": config['Ferramentas'].get('posicao', 'false') == 'true',
        "n_rois": config.getint('Ferramentas', 'n_rois')
    })



# Endpoint POST para alterar os parâmetros de configuração
@app.route('/coord_ref', methods=['POST'])
def update_coordRef_config():
    
    data = request.json

    programa = data.get('programa')  # Pegando o valor de 'programa' do JSON
    roi = data.get('roi') 
    PATH = f'config_{programa}.ini'
    config = load_config(PATH)

    
    if 'x_ref' in roi:
        config['Referencia']['x_ref'] = str(roi['x_ref'])
    if 'y_ref' in roi:
        config['Referencia']['y_ref'] = str(roi['y_ref'])
    if 'x_interes' in roi:
        config['Referencia']['x_interes'] = str(roi['x_interes'])
    if 'y_interes' in roi:
        config['Referencia']['y_interes'] = str(roi['y_interes'])
    if 'width_interes' in roi:
        config['Referencia']['width_interes'] = str(roi['width_interes'])
    if 'height_interes' in roi:
        config['Referencia']['height_interes'] = str(roi['height_interes'])
 

    # Salva as alterações no arquivo INI
    save_config(PATH,config)
    return jsonify({"message": "Configuração atualizada com sucesso!"})

@app.route('/template', methods=['POST'])
def definirTemplate():
    
    
    data = request.json  # Captura o JSON enviado no corpo da requisição
    programa = data.get('programa')  # Acessa o valor do programa
    PATH= f'config_{programa}.ini'
    # Carrega as configurações atuais
    config = load_config(PATH)
    
    # Acessa os dados da ROI
    roi = data.get('roi', {})

    # Verifica se as chaves esperadas estão no JSON recebido
    if not all(k in roi for k in ["x", "y", "width", "height"]):
        return jsonify({"error": "Dados incompletos. Faltam campos obrigatórios."}), 400

    # Armazena os valores recebidos em variáveis globais
    x = roi["x"]
    y = roi["y"]
    width = roi["width"]
    height = roi["height"]
    
    # Processar a imagem como no código original
    response = requests.get(url)
        #response = requests.capture_frame()
    if response.status_code == 200:
            nparr = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Realizar o recorte
    recorte = img[y:y+height, x:x+width]

    # Salvar o recorte como PNG
    cv2.imwrite(f"template_{programa}.png", recorte)

    roi_regiao_template = []

    x_interes = int(config['Referencia']['x_interes'])
    y_interes = int(config['Referencia']['y_interes'])
    w_interes = int(config['Referencia']['width_interes'])
    h_interes = int(config['Referencia']['height_interes'])

    
    roi_regiao_template.append((x_interes, y_interes, w_interes, h_interes))

    template = cv2.imread(f'template_{programa}.png', 0)

    frame = cv2.imread(f'ref_programa{programa}.png', 0)
    
    # salvar referencia.
    ref_coord = Localizador.coord_ref(recorte, roi_regiao_template[0], img)
    
    # Atualizar o arquivo de configuração com as coordenadas de referência
    config['Referencia']['x_ref'] = str(ref_coord[0])
    config['Referencia']['y_ref'] = str(ref_coord[1])
    config['Referencia']['width_ref'] = str(width)
    config['Referencia']['height_ref'] = str(height)
    
    # Salvar as alterações no arquivo INI
    save_config(PATH,config)

    return jsonify({"message": "Configuração atualizada com sucesso!", "dados_recebidos": data})

# Endpoint POST para acessar os parâmetros de configuração
@app.route('/coord_ref', methods=['GET'])
def get_coordRef_config():
    programa = request.args.get('programa')  # Usando query string para GET
    

    PATH = f'config_{programa}.ini'

    # Carrega o arquivo de configuração
    config = load_config(PATH)
  

    # Retorna os valores de cada configuração
    return jsonify({
         "x_ref": config.getint('Referencia', 'x_ref', fallback=0),
         "y_ref": config.getint('Referencia', 'y_ref',fallback=0),
        "x_interes": config.getint('Referencia', 'x_interes', fallback=0),
        "y_interes": config.getint('Referencia', 'y_interes', fallback=0),
        "width_interes": config.getint('Referencia', 'width_interes', fallback=0),
        "height_interes": config.getint('Referencia', 'height_interes', fallback=0)
    })




# Endpoint GET para retornar a imagem
@app.route('/image', methods=['GET'])
def get_image():
    if os.path.exists(IMAGE_PATH):
        return send_file(IMAGE_PATH, mimetype='image/png')
    else:
        return jsonify({"error": "Imagem não encontrada"}), 404
    
@app.route('/video_feed')
def video_feed():
    """ Rota que transmite o vídeo ao vivo via HTTP """
    def generate_frames():
        while True:
            with lock:
                if last_frame is not None:
                    _, buffer = cv2.imencode('.jpg', last_frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)  # Taxa de atualização do stream
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Rota para obter os limites de uma ROI
@app.route('/get_limits/<roi>', methods=['GET'])
def get_limits(roi):

    
    programa = request.args.get('programa')  # Usando query string para GET
    

    PATH = f'config_{programa}.ini'

    # Carrega o arquivo de configuração
    config = load_config(PATH)
  
    if roi not in config:
        return jsonify({"error": "ROI não encontrada"}), 404
    
    limits = {
        "textura_tolerancia": int(config[roi]['textura_tolerancia']),
        "pixel_tolerancia": int(config[roi]['pixel_tolerancia']),
        "threshold_cor": int(config[roi]['threshold_cor']),
        "ponto_tolerancia": int(config[roi]['ponto_tolerancia'])

    }
    return jsonify(limits)

# Rota para alterar os limites de uma ROI
@app.route('/set_limits/<roi>', methods=['POST'])
def set_limits(roi):

     # Obtém os limites enviados na requisição
    data = request.json
    programa = data.get('programa')  # Acessa o valor do programa
    PATH = f'config_{programa}.ini'
    config = load_config(PATH)
    if roi not in config:
        return jsonify({"error": "ROI não encontrada"}), 404

   
    required_keys = ["textura_tolerancia", "pixel_tolerancia"]
    if not all(key in data for key in required_keys):
        return jsonify({"error": "Faltando parâmetros necessários"}), 400
    
    # Atualiza os limites no arquivo
    config[roi]['textura_tolerancia'] = str(data["textura_tolerancia"])
    config[roi]['pixel_tolerancia'] = str(data["pixel_tolerancia"])
    config[roi]['threshold_cor'] = str(data["threshold_cor"])
    config[roi]['ponto_tolerancia'] = str(data["ponto_tolerancia"])
   

    # Escreve as alterações no arquivo
    save_config(PATH,config)

    return jsonify({"message": "Limites atualizados com sucesso!"}), 200

@app.route('/list_cameras', methods=['GET'])
def list_cameras():
    """ Lista as câmeras Basler conectadas ao sistema """
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()

    cameras = [
        {"index": i, "name": device.GetFriendlyName(), "serial": device.GetSerialNumber()}
        for i, device in enumerate(devices)
    ]

    return jsonify({"cameras": cameras})

@app.route('/set_camera', methods=['POST'])
def set_camera():
    """ Define o tipo de câmera e o número de série (se for Basler) """
    data = request.json
    camera_type = data.get("tipo")
    serial_number = data.get("serial_number")

    if not camera_type:
        return jsonify({"error": "O campo 'tipo' é obrigatório"}), 400

    if camera_type.lower() == "basler":
        available_serials = get_available_basler_cameras()
        if serial_number not in available_serials:
            return jsonify({"error": "Número de série inválido ou câmera não encontrada", "disponíveis": available_serials}), 400

    save_camera_config(camera_type, serial_number)
    return jsonify({"message": "Configuração salva com sucesso", "tipo": camera_type, "serial_number": serial_number})

@app.route('/reboot', methods=['POST'])
def reboot_system():
    """ Rota para reiniciar o Raspberry Pi """
    try:
        os.system("sudo reboot")
        return jsonify({"message": "Raspberry Pi reiniciando..."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/count_nok_files/<int:program_number>', methods=['GET'])
def count_nok_files(program_number):
    # Padrão esperado: "ref_programa{numero}_NOK" seguido de opcional "_X" e extensão .png
    pattern = re.compile(rf'^ref_programa{program_number}_NOK(_\d+)?\.png$')
    
    # Lista todos os arquivos no diretório atual (ou ajuste o caminho)
    files = os.listdir('.')  # Ou defina um diretório específico, ex: '/path/to/files'
    
    # Filtra os arquivos que correspondem ao padrão
    matching_files = [file for file in files if pattern.match(file)]
    
    # Retorna a contagem em JSON
    return jsonify({
        "program_number": program_number,
        "count": len(matching_files),
        "matching_files": matching_files  # Opcional: listar os arquivos encontrados
    })

@app.route('/count_ok_files/<int:program_number>', methods=['GET'])
def count_ok_files(program_number):
    # Padrão esperado: "ref_programa{numero}_NOK" seguido de opcional "_X" e extensão .png
    pattern = re.compile(rf'^ref_programa{program_number}_OK(_\d+)?\.png$')
    
    # Lista todos os arquivos no diretório atual (ou ajuste o caminho)
    files = os.listdir('.')  # Ou defina um diretório específico, ex: '/path/to/files'
    
    # Filtra os arquivos que correspondem ao padrão
    matching_files = [file for file in files if pattern.match(file)]
    
    # Retorna a contagem em JSON
    return jsonify({
        "program_number": program_number,
        "count": len(matching_files),
        "matching_files": matching_files  # Opcional: listar os arquivos encontrados
    })

@app.route('/cadastrar', methods=['POST'])
def cadastrar():

    pasta = PASTA_ROSTOS
    if not os.path.exists(pasta):
        os.makedirs(pasta)

    # Verifica se é JSON
    if not request.is_json:
        return jsonify({"erro": "Content-Type deve ser application/json"}), 400
        
    data = request.get_json()
    nome = data.get('nome', '').strip()
    
    if not nome:
        return jsonify({"erro": "Nome é obrigatório"}), 400
    
    global last_frame
    if last_frame is None:
        return jsonify({"erro": "Nenhum frame disponível da câmera"}), 400
    
    try:
        imagem = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        rostos = face_recognition.face_encodings(imagem)
        
        if rostos:
           # np.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(nome) + ".npy"), rostos[0])
            np.save(os.path.join(pasta, nome + ".npy"), rostos[0])
            return jsonify({
                "sucesso": True,
                "mensagem": f"Rosto de {nome} cadastrado com sucesso!"
            })
        
        return jsonify({"erro": "Nenhum rosto detectado no frame atual"}), 400
        
    except Exception as e:
        return jsonify({"erro": f"Erro ao processar frame: {str(e)}"}), 500

@app.route('/status.json')
def status_json():
    config = configparser.ConfigParser()
    config.read('status.ini')

    if 'STATUS' not in config:
        return jsonify({'erro': 'Seção STATUS não encontrada'}), 500

    s = config['STATUS']
    data = {
        'programa': int(s.get('programa', 0)),
        'nome_programa': s.get('nome_programa', 'desconhecido'),
        'n_rois': int(s.get('n_rois', 0)),
        'status_geral': s.getboolean('status_geral', fallback=False),
        'tempo_execucao': float(s.get('tempo_execucao', 0.0)),
        'imagem': 'imagem_status.jpg'
    }

    # Adiciona os parciais (ROI)
    for i in range(1, 13):
        key = f'parcial_p{i}'
        data[key] = s.getboolean(key, fallback=False)

    return jsonify(data)

# Servir imagem diretamente se estiver na mesma pasta
@app.route('/imagem_status.jpg')
def imagem_status():
    return send_file('imagem_status.jpg', mimetype='image/jpeg')



import time  # Certifique-se que está importado no topo do arquivo

@app.route('/reconhecer', methods=['POST'])
def reconhecer_api():
    """Endpoint com loop de 15 segundos para reconhecimento"""
    global last_frame
    
    if last_frame is None:
        return jsonify({"erro": "Nenhum frame disponível da câmera"}), 400
    
    try:
        rostos_conhecidos, nomes = carregar_rostos()
        if not rostos_conhecidos:
            return jsonify({"erro": "Nenhum rosto cadastrado ainda"}), 400
        
        start_time = time.time()
        timeout = 15  # segundos
        resultados = []

        while time.time() - start_time < timeout:
            if last_frame is None:
                continue
                
            frame_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
            rostos = face_recognition.face_locations(frame_rgb)
            codificacoes = face_recognition.face_encodings(frame_rgb, rostos)

            for (top, right, bottom, left), cod in zip(rostos, codificacoes):
                distancias = face_recognition.face_distance(rostos_conhecidos, cod)
                min_dist = np.min(distancias)
                
                if min_dist < 0.5:  # Se reconheceu alguém
                    index = np.argmin(distancias)
                    resultados.append({
                        "posicao": {"top": top, "right": right, "bottom": bottom, "left": left},
                        "nome": nomes[index],
                        "confianca": float(1 - min_dist),
                        "tempo_de_reconhecimento": time.time() - start_time
                    })
                    return jsonify({
                        "sucesso": True,
                        "resultados": resultados,
                        "mensagem": "Rosto reconhecido dentro do tempo limite"
                    })
            
            time.sleep(0.1)  # Pequena pausa para não sobrecarregar

        return jsonify({
            "sucesso": False,
            "resultados": resultados,
            "mensagem": "Tempo limite atingido sem reconhecimento"
        })

    except Exception as e:
        return jsonify({
            "sucesso": False,
            "erro": str(e)
        }), 500
    
# Iniciar o servidor Flask
if __name__ == '__main__':
    if(cam_tipo == "webcam"):
        initialize_camera()  # Inicializa a câmera antes de iniciar o servidor
    else:
       initialize_camera_basler()  # Inicia a câmera antes de rodar o servidor Flask
    app.run(host='0.0.0.0', port=6001)

