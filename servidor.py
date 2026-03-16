from sys import platform
from flask import Flask,abort, request, Response, jsonify, send_file,stream_with_context, render_template, render_template_string

import configparser
import cv2
import threading
import os
from flask_cors import CORS
import requests
import json
#import face_recognition
import numpy as np
import time
from datetime import datetime

import torch
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
import urllib.parse
import treinar_modelo
import treinar_modelo_new
import monitorar_movimentos

import logging
import subprocess
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras import mixed_precision
import datetime

#from db import get_db

# Pasta das imagens de referência para RNA 
IMG_DIR = os.path.join(os.path.dirname(__file__), 'static', 'imagens')

from pypylon import pylon
from werkzeug.utils import secure_filename
app = Flask(__name__)


modo_treino = False

# Variáveis globais
camera_ociosa = False
ultima_atividade = None
monitorando = True
capture_thread_running = True
PASTA_ROSTOS = 'rostos_cadastrados_cpu'
# Dicionário de controle por câmera
cameras = {}
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

def check_camera_ociosa_webcam(): #obs: pedente 
    global camera_ociosa, ultima_atividade, monitorando, capture_thread_running
    
    while True:
        tempo_atual = datetime.now()
        
        if ultima_atividade is not None and monitorando:
            tempo_ocioso = (tempo_atual - ultima_atividade).total_seconds()
            
            if tempo_ocioso > 60:  # 15 minutos em segundos
                print("Passou mais de 15 minutos sem atividade. Desligando a câmera.")
                camera_ociosa = True
                
                # Primeiro para a thread de captura
                capture_thread_running = False

                
                try:
                    if(cam_tipo == "basler"):
                       camera.Close()
                    else:
                       camera.release()
                except:
                    print("Erro ao liberar câmera")
                monitorando = False  # Para o loop de monitoramento
                
        time.sleep(10)  # Verifica a cada 10 segundos
          
def check_camera_ociosa_loop():
    while True:
        for serial, cam_data in cameras.items():
            if cam_data["capture_thread_running"] and not cam_data.get("ociosa", False):
                tempo_ocioso = (datetime.now() - cam_data["ultima_atividade"]).total_seconds()
                if tempo_ocioso > 70:  # 15 minutos
                    print(f"[{serial}] Inativa por 15min. Encerrando câmera.")
                    cam_data["capture_thread_running"] = False
                    cam_data["ociosa"] = True

                    try:
                        if cam_data["camera"]:
                            if cam_data["camera"].IsGrabbing():
                                cam_data["camera"].StopGrabbing()
                            cam_data["camera"].Close()
                    except Exception as e:
                        print(f"[{serial}] Erro ao fechar por ociosidade: {e}")

        time.sleep(10)
#threading.Thread(target=check_camera_ociosa_loop, daemon=True).start()

# Criando e iniciando a thread(nao usado temporariamente)
#thread_cam_ociosa = threading.Thread(target=check_camera_ociosa)
#thread_cam_ociosa.daemon = True  # Encerra a thread quando o programa principal terminar
#thread_cam_ociosa.start() 



def get_serial():
    try:
        serial = None

        # Primeiro tenta o caminho do Jetson Orin / Jetson Nano
        if os.path.exists("/sys/firmware/devicetree/base/serial-number"):
            with open("/sys/firmware/devicetree/base/serial-number", "r") as f:
                serial = f.read().strip()
                if serial:
                    return serial

        # Se não achar (por exemplo, se for Raspberry Pi), tenta o /proc/cpuinfo
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("Serial"):
                    return line.strip().split(":")[1].strip()

    except Exception as e:
        print(f"Erro ao obter serial: {e}")
        return None

# ✅ Substitua com o serial autorizado do seu Jetson (pegue com 'cat /sys/firmware/devicetree/base/serial-number')
serial_autorizado = None #"2bfa9f2a43fe3e38"

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

def salvar_imagem_no_pendrive(camera,image, pendrive_path='/media/usb'):
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
    if (camera == 1):
        pasta_img = f"cam{camera}_log_imagens"
    elif(camera == 2):
        pasta_img = f"cam{camera}_log_imagens"
    else:
        pasta_img = "log_imagens"

    caminho_pasta = os.path.join(pendrive_path, pasta_img, ano, mes, dia)
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

# Obter os caminhos das câmeras webcam e basler
cam_path = config.get("CAMERAS", "cam")
cam_1 = config.get("CAMERAS", "cam_1") #comando para listar as cameras disponiveis:  ls -l /dev/v4l/by-id/
cam_2 = config.get("CAMERAS", "cam_2") #comando para listar as cameras disponiveis:  ls -l /dev/v4l/by-id/
cam_3 = config.get("CAMERAS", "cam_3") #comando para listar as cameras disponiveis:  ls -l /dev/v4l/by-id/

cam1_encoded = urllib.parse.quote(cam_1, safe='')
cam2_encoded = urllib.parse.quote(cam_2, safe='')
cam3_encoded = urllib.parse.quote(cam_3, safe='')

cam_tipo = config.get("CAMERAS", "tipo")
cam_basler_conexao = config.get("CAMERAS", "communication")
serial_number_1 = config.get("CAMERAS","serial_number_basler_1")
serial_number_2 = config.get("CAMERAS","serial_number_basler_2")
serial_number_3 = config.get("CAMERAS","serial_number_basler_3")


# Lista de câmeras esperadas (pode conter só uma)
CAMERA_SERIALS = [serial_number_1, serial_number_2,serial_number_3]
cameras_web = {}
CAMERA_IDS_WEB = [cam_1, cam_2,cam_3]  # cam_1 e cam_2 são índices (ex: 0, 1)


# Variável global para a câmera
camera = None
lock = threading.Lock()  # Para evitar concorrência nos frames
last_frame = None


# Configuração de tamanho da imagem
TARGET_WIDTH = 1024  # Largura desejada antiga 640
TARGET_HEIGHT = 768  # Altura desejada        480
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

def initialize_all_webcams():
    """ Inicializa watchdogs para todas as webcams configuradas """
    for cam_id in CAMERA_IDS_WEB:
        cameras_web[cam_id] = {
            "camera": None,
            "last_frame": None,
            "lock": threading.Lock(),
            "capture_thread_running": False,
            "ultima_atividade": datetime.datetime.now(),
            "ociosa": False
        }
        threading.Thread(target=webcam_watchdog_loop, args=(cam_id,), daemon=True).start()

def open_camera(src):
    # Câmera IP (RTSP)
    import platform
    if src.startswith("rtsp://"):
        return cv2.VideoCapture(src, cv2.CAP_FFMPEG)

    # Webcam / CSI
    if src.startswith("/dev/"):
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    if platform.system().lower() == "windows":
        cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    # Fallback (segurança)
    return cv2.VideoCapture(src)

def webcam_watchdog_loop(cam_id):
    while True:
        cam_data = cameras_web[cam_id]

        if cam_data.get("ociosa", False):
            time.sleep(3)
            continue

        try:
            print(f"[WATCHDOG WEBCAM {cam_id}] Iniciando conexão...")
            #camera = cv2.VideoCapture(cam_id)
            #camera = cv2.VideoCapture(cam_id, cv2.CAP_FFMPEG)
            #camera = cv2.VideoCapture(cam_id, cv2.CAP_V4L2) 
            camera = open_camera(cam_id)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            camera.set(cv2.CAP_PROP_FPS, 16)

            if not camera.isOpened():
                print(f"[WATCHDOG WEBCAM {cam_id}] Não foi possível abrir a câmera.")
                time.sleep(3)
                continue

            cam_data["camera"] = camera
            cam_data["capture_thread_running"] = True

            webcam_loop(cam_id)

        except Exception as e:
            print(f"[WATCHDOG WEBCAM {cam_id}] Erro: {e}")
            cam_data["camera"] = None
            cam_data["capture_thread_running"] = False

        time.sleep(3)


def webcam_loop(cam_id):
    cam_data = cameras_web[cam_id]
    camera = cam_data["camera"]

    while cam_data["capture_thread_running"] and camera.isOpened():
        try:
            ret, frame = camera.read()
            if ret:
                with cam_data["lock"]:
                    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
                    cam_data["last_frame"] = frame.copy()
                    # cam_data["ultima_atividade"] = datetime.now()  # deixe o controle externo

            #time.sleep(0.03)

        except Exception as e:
            print(f"[CAPTURA WEBCAM {cam_id}] Erro: {e}")
            cam_data["capture_thread_running"] = False
            break

    print(f"[CAPTURA WEBCAM {cam_id}] Thread finalizada.")

def get_latest_webcam_frame(cam_id):
    # Verifica se existe uma chave que termine com o cam_id fornecido
    for key in cameras_web:
        if key.endswith(cam_id.strip()):
            cam_id = key
            break
    else:
        # Se nenhum match for encontrado
        return None

    # Agora acessa normalmente
    if cam_id in cameras_web:
        with cameras_web[cam_id]["lock"]:
            return cameras_web[cam_id]["last_frame"]
    return None

def get_latest_webcam_frame_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        nparr = np.frombuffer(response.content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame

def initialize_all_cameras_basler_usb():
    """ Inicializa watchdogs para todas as câmeras configuradas """
    for serial in CAMERA_SERIALS:
        cameras[serial] = {
            "camera": None,
            "last_frame": None,
            "lock": threading.Lock(),
            "capture_thread_running": False,
            "ultima_atividade": datetime.datetime.now(),  # Nova chave
            "ociosa": False  # Flag de ociosidade
        }
        threading.Thread(target=camera_watchdog_loop, args=(serial,), daemon=True).start()
        

def initialize_all_cameras_basler_gige():
    """ Inicializa watchdogs para todas as câmeras configuradas """
    for serial in CAMERA_SERIALS:
        cameras[serial] = {
            "camera": None,
            "last_frame": None,
            "lock": threading.Lock(),
            "capture_thread_running": False,
            "ultima_atividade": datetime.datetime.now(),  # Nova chave
            "ociosa": False  # Flag de ociosidade
        }
        #threading.Thread(target=camera_watchdog_loop, args=(serial,), daemon=True).start()
        threading.Thread(target=camera_watchdog_loop_gige, args=(serial,), daemon=True).start()

def camera_watchdog_loop(serial_number):
    """ Watchdog que mantém a câmera ativa e reconecta se necessário """
    tl_factory = pylon.TlFactory.GetInstance()

    while not modo_treino:

        if modo_treino:
            time.sleep(1)
            continue
        cam_data = cameras.get(serial_number)
        if cam_data and cam_data.get("ociosa", False):
            # Se a câmera estiver ociosa, não tenta reconectar
            time.sleep(3)
            continue
        try: 
                devices = tl_factory.EnumerateDevices()
                camera_device = None

                for device in devices:
                    if device.GetSerialNumber() == serial_number:
                        camera_device = device
                        break

                if camera_device is None:
                    print(f"[WATCHDOG {serial_number}] Câmera não encontrada. Aguardando...")
                    time.sleep(3)
                    continue

                print(f"[WATCHDOG {serial_number}] Câmera encontrada. Iniciando...")
                camera = pylon.InstantCamera(tl_factory.CreateDevice(camera_device))
                camera.Open()
                camera.MaxNumBuffer.SetValue(3)
                camera.OutputQueueSize.SetValue(1)
                camera.AcquisitionMode.Value = "Continuous"
                camera.AcquisitionFrameRateEnable.Value = True
                camera.AcquisitionFrameRate.Value = 16.0
                camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

                converter = pylon.ImageFormatConverter()
                converter.OutputPixelFormat = pylon.PixelType_BGR8packed
                converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

                cameras[serial_number]["camera"] = camera
                cameras[serial_number]["capture_thread_running"] = True

                camera_loop_basler(serial_number, converter)

        except Exception as e:
            print(f"[WATCHDOG {serial_number}] Erro: {e}")
            try:
                cam = cameras[serial_number]["camera"]
                if cam and cam.IsGrabbing():
                    cam.StopGrabbing()
                if cam:
                    cam.Close()
            except Exception as e_close:
                print(f"[WATCHDOG {serial_number}] Erro ao fechar câmera: {e_close}")

            cameras[serial_number]["camera"] = None
            cameras[serial_number]["capture_thread_running"] = False
            time.sleep(3)  # Espera e tenta novamente

def set_node_if_exists(camera, node_name, value):
    try:
        node = camera.GetNodeMap().GetNode(node_name)
        if node and node.IsWritable():
            node.SetValue(value)
            return True
    except:
        pass
    return False

def camera_watchdog_loop_gige(camera_id):
    tl_factory = pylon.TlFactory.GetInstance()

    while not modo_treino:

        cam_data = cameras.get(camera_id)
        if cam_data and cam_data.get("ociosa", False):
            time.sleep(3)
            continue

        try:
            devices = tl_factory.EnumerateDevices()
            camera_device = None

            for device in devices:
                if device.GetDeviceClass() == "BaslerGigE":
                    if device.GetUserDefinedName() == camera_id:
                        camera_device = device
                        break

            if camera_device is None:
                print(f"[WATCHDOG {camera_id}] Câmera GigE não encontrada.")
                time.sleep(3)
                continue

            print(f"[WATCHDOG {camera_id}] Câmera GigE encontrada. Iniciando...")

            camera = pylon.InstantCamera(
                tl_factory.CreateDevice(camera_device)
            )

            camera.Open()
            try:
                camera.TriggerSelector.Value = "FrameStart"
                camera.TriggerMode.Value = "Off"
            except:
                pass
            camera.AcquisitionMode.Value = "Continuous"

            set_node_if_exists(camera, "TriggerSelector", "FrameStart")
            set_node_if_exists(camera, "TriggerMode", "Off")

            set_node_if_exists(camera, "AcquisitionFrameRateEnable", True)
            set_node_if_exists(camera, "AcquisitionFrameRate", 16.0)

            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            cameras[camera_id]["camera"] = camera
            cameras[camera_id]["capture_thread_running"] = True

            camera_loop_basler_gige(camera_id, converter)

        except Exception as e:
            print(f"[WATCHDOG {camera_id}] Erro: {e}")

            try:
                cam = cameras[camera_id]["camera"]
                if cam and cam.IsGrabbing():
                    cam.StopGrabbing()
                if cam:
                    cam.Close()
            except:
                pass

            cameras[camera_id]["camera"] = None
            cameras[camera_id]["capture_thread_running"] = False
            time.sleep(3)



def camera_loop_basler(serial_number, converter):
    """ Captura frames continuamente da câmera especificada """
    cam_data = cameras[serial_number]
    camera = cam_data["camera"]

    #while cam_data["capture_thread_running"] and camera.IsGrabbing():
    while (
        cam_data["capture_thread_running"]
        and camera
        and camera.IsGrabbing()
        and not modo_treino
    ):

        try:
            grabResult = camera.RetrieveResult(500, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                frame = image.GetArray()

                frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

                with cam_data["lock"]:
                    cam_data["last_frame"] = frame
                    #cam_data["ultima_atividade"] = datetime.now() #*

            grabResult.Release()
            time.sleep(0.01)

        except Exception as e:
            print(f"[CAPTURA {serial_number}] Erro ao capturar: {e}")
            cam_data["capture_thread_running"] = False
            break

    print(f"[CAPTURA {serial_number}] Thread finalizada.")

def camera_loop_basler_gige(serial_number, converter):
    """ Captura frames continuamente da câmera especificada """
    cam_data = cameras[serial_number]
    camera = cam_data["camera"]

    #while cam_data["capture_thread_running"] and camera.IsGrabbing():
    while (
        cam_data["capture_thread_running"]
        and camera
        and camera.IsGrabbing()
        and not modo_treino
    ):

        try:
            grabResult = camera.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                frame = image.GetArray()

                frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

                with cam_data["lock"]:
                    cam_data["last_frame"] = frame
                    #cam_data["ultima_atividade"] = datetime.now() #*

            grabResult.Release()
            time.sleep(0.01)

        except Exception as e:
            print(f"[CAPTURA {serial_number}] Erro ao capturar: {e}")
            cam_data["capture_thread_running"] = False
            break

    print(f"[CAPTURA {serial_number}] Thread finalizada.")


def parar_todas_cameras():
    global cameras
    for serial, cam_data in cameras.items():
        cam_data["ociosa"] = True
        cam_data["capture_thread_running"] = False
        
        cam = cam_data.get("camera")
        if cam:
            try:
                if cam.IsGrabbing():
                    cam.StopGrabbing()
                cam.Close()
            except:
                pass

        cam_data["camera"] = None


def atualizar_atividade_camera(serial_number): #*
    if serial_number in cameras:
        cameras[serial_number]["ultima_atividade"] = datetime.now()

def reativar_camera(serial_number): #*
    cam_data = cameras.get(serial_number)
    if not cam_data:
        
        return f"[{serial_number}] Câmera não registrada."

    if cam_data["ociosa"]:
        cam_data["ociosa"] = False
        threading.Thread(target=camera_watchdog_loop, args=(serial_number,), daemon=True).start()
        return f"[{serial_number}] Reativação iniciada."
    else:
        return f"[{serial_number}] Câmera não estava ociosa."


def get_latest_frame(serial_number):
    """ Retorna o último frame capturado da câmera especificada """
    if serial_number in cameras:
        with cameras[serial_number]["lock"]:
            return cameras[serial_number]["last_frame"]
    return None
def get_png_frame():
    """
    Retorna um frame a partir de um arquivo PNG.
    Pode usar o serial_number para escolher a imagem.
    """

    # Exemplo de caminho da imagem
    image_path = "teste_sve.png"

    if not os.path.exists(image_path):
        print(f"Imagem não encontrada: {image_path}")
        return None

    frame = cv2.imread(image_path)

    if frame is None:
        print("Erro ao carregar a imagem.")
        return None

    return frame

@app.route('/upload_png', methods=['POST'])
def upload_png():

    if 'file' not in request.files:
        return jsonify({"erro": "Nenhum arquivo enviado"}), 400

    file = request.files['file']

    if file.filename == "":
        return jsonify({"erro": "Nome de arquivo vazio"}), 400

    if not file.filename.lower().endswith(".png"):
        return jsonify({"erro": "Apenas arquivos PNG são permitidos"}), 400

    caminho = os.path.join("", "teste_sve.png")

    # salva sobrescrevendo
    file.save(caminho)

    return jsonify({
        "status": "ok",
        "arquivo_salvo": caminho
    })



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
def desenhar_roi_old(imagem, x, y, w, h, status, posicao, forma="retangulo"):
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


import cv2

# Função para desenhar a ROI (retângulo ou círculo) e escrever a label
def desenhar_roi(imagem, x, y, w, h, status, posicao, forma="retangulo", ativar_label = True, label="ROI"):
    """
    Desenha uma ROI na imagem e escreve a label ao lado dela.
    
    :param imagem: Imagem onde a ROI será desenhada.
    :param x, y: Coordenadas do canto superior esquerdo (retângulo) ou centro (círculo).
    :param w, h: Largura e altura da ROI (para retângulo) ou raio (para círculo).
    :param status: Status da ROI (True ou False).
    :param posicao: Define cor especial (exemplo: azul).
    :param forma: Forma da ROI ("retangulo" ou "circulo").
    :param label: Texto da label a ser desenhada.
    """

    if posicao:
        cor_roi = (255, 0, 0)  # Azul
    else:
        cor_roi = (0, 255, 0) if status else (0, 0, 255)  # Verde ou vermelho

    # Desenhar a ROI
    if forma == "retangulo":
        cv2.rectangle(imagem, (x, y), (x + w, y + h), cor_roi, 1)
        # posição do texto: à direita do retângulo
        text_x = x + w + 5
        text_y = y + int(h / 2)
    elif forma == "circulo":
        centro = (x, y)
        raio = w
        cv2.circle(imagem, centro, raio, cor_roi, 1)
        # posição do texto: à direita do círculo
        text_x = x + raio + 5
        text_y = y
    else:
        raise ValueError("Forma não suportada. Use 'retangulo' ou 'circulo'.")
    if(ativar_label):
        # Desenhar a label
        cv2.putText(
            imagem,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,        # escala da fonte
            (0, 255, 255),  # amarelo
            1,
            cv2.LINE_AA
    )

API_TOKEN = "123#"

LOG_PATH = os.path.expanduser("~/restart_jetson.log")

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def reiniciar_ms_vision(delay=5):
    logging.info(f"Reboot autorizado. Reiniciando em {delay} segundos")
    time.sleep(delay)
    ps1_path = r"reboot.ps1"
    subprocess.run(
        [
            "powershell.exe",
            "-ExecutionPolicy", "Bypass",
            "-NoProfile",
            "-File", ps1_path
        ],
        creationflags=subprocess.CREATE_NO_WINDOW
    )

@app.route("/reboot", methods=["POST"])
def reboot():
   #token = request.headers.get("Authorization")

    #if token != f"Bearer {API_TOKEN}":
     #   logging.warning("Tentativa de reboot não autorizada")
      #  return jsonify({"error": "Não autorizado"}), 401

    threading.Thread(target=reiniciar_ms_vision, args=(5,), daemon=True).start()

    return jsonify({
        "status": "ok",
        "message": "Reboot agendado para 5 segundos"
    }), 200

############################################# REDE NEURAL ARTIFICIAL #################################

mixed_precision.set_global_policy('mixed_float16')

MODELOS = {}

def carregar_imagem(imagem, coordenadas):
    x, y, w, h = coordenadas
    roi = imagem[y:y+h, x:x+w]
    roi = cv2.resize(roi, (28,28)).astype("float32") / 255.0
    return roi.reshape(1,28,28,1)

def carregar_modelo(camera, programa, roi):
    chave = f"{camera}_{programa}_{roi}"
    if chave not in MODELOS:
        caminho = f"modelo_cam{camera}_prog{programa}_ROI{roi}.keras"
        MODELOS[chave] = load_model(caminho)
        print(f"✅ Modelo {chave} carregado na GPU.")
    return MODELOS[chave]

def predizer_frame(camera, programa, roi, gray, coordenadas):
    model = carregar_modelo(camera, programa, roi)
    img = carregar_imagem(gray, coordenadas)
    pred = model.predict(img, verbose=0)
    return bool(pred[0][0] >= 0.5)

def predizer_frame_new(camera, programa, roi, frame, coordenadas, threshold=0.5):
    """
    Predição OK / NOK usando TensorFlow + Transfer Learning

    :param camera: id da câmera
    :param programa: id do programa
    :param roi: índice da ROI
    :param frame: frame OpenCV (BGR)
    :param coordenadas: (x, y, w, h)
    :param threshold: limiar de decisão
    :return: True = OK | False = NOK
    """

    # Carrega modelo treinado
    model = carregar_modelo(camera, programa, roi)

    x, y, w, h = coordenadas

    # --- Recorte da ROI ---
    roi_img = frame[y:y+h, x:x+w]

    if roi_img.size == 0:
        return False  # fail-safe

    # --- Pré-processamento (IGUAL AO TREINO) ---
    roi_img = cv2.resize(roi_img, (224, 224))
    roi_img = roi_img.astype("float32") / 255.0

    # Batch dimension
    roi_img = np.expand_dims(roi_img, axis=0)

    # --- Predição ---
    prob_ok = float(model.predict(roi_img, verbose=0)[0][0])

    # --- Decisão ---
    return prob_ok >= threshold
######################################################## TREINO IA ########################################################

from training_state import read_status


@app.route("/training_status")
def training_status_api():
    return jsonify(read_status())

from training_state import write_status

def executar_treino_async(camera, programa, tipo_neural):
    global modo_treino

    try:
        if tipo_neural == "yolo":
            treinar_modelo_new.gerar_modelo(
                camera=camera,
                programa=programa
            )
        elif tipo_neural == "resnet":
            treinar_modelo_new.gerar_modelo_resnet(camera, programa)
        elif tipo_neural == "resnet50":
            treinar_modelo_new.gerar_modelo_resnet50(camera, programa)
        elif tipo_neural == "mobilenetv2":
            treinar_modelo_new.gerar_modelo_mobilenet(camera, programa)

        #  FINALIZA COM SUCESSO
        write_status({
            "running": False,
            "epoch": 100,
            "total_epochs": 100,
            "progress": 100,
            "mensagem": "Treinamento concluído"
        })

    except Exception as e:
        write_status({
            "running": False,
            "epoch": 0,
            "total_epochs": 0,
            "progress": 0,
            "mensagem": f"Erro: {str(e)}"
        })

    finally:
        modo_treino = False
        if cam_tipo == "basler":
            if cam_basler_conexao == "usb":
                initialize_all_cameras_basler_usb()
            else:
                initialize_all_cameras_basler_gige()
        elif cam_tipo == "webcam":
            initialize_all_webcams()
        else:
            print("Modo Simulacao: Nenhuma câmera real será inicializada. O servidor servirá imagens estáticas para teste.")  






from threading import Thread
from training_state import write_status
EPOCHS =100
@app.route("/treinar", methods=["POST"])
def treinar():
    global modo_treino

    data = request.get_json()
    programa = data.get("programa")
    camera = data.get("camera")

    config = configparser.ConfigParser()
    config.read(f'config_{camera}_{programa}.ini')
    tipo_neural = str(config['Modelos']['tipo_rede_neural_artificial'])

    modo_treino = True

    # ✅ INICIALIZA STATUS NO ARQUIVO
    write_status({
        "running": True,
        "epoch": 0,
        "total_epochs": EPOCHS,
        "progress": 0,
        "mensagem": "Iniciando treinamento..."
    })

    parar_todas_cameras()
    time.sleep(2)

    import torch
    torch.cuda.empty_cache()

    # ✅ THREAD DE TREINO
    Thread(
        target=executar_treino_async,
        args=(camera, programa, tipo_neural),
        daemon=True
    ).start()

    # ✅ RESPONDE IMEDIATAMENTE
    return jsonify({
        "status": "sucesso",
        "mensagem": "Treinamento iniciado"
    })
########################################################################################################

url = "http://127.0.0.1:6001/capture"

url_cam1 = f"http://127.0.0.1:6001/capture/{serial_number_1}"
url_cam2 = f"http://127.0.0.1:6001/capture/{serial_number_2}"
url_cam3 = f"http://127.0.0.1:6001/capture/{serial_number_2}"


url_web_cam1 = f"http://127.0.0.1:6001/capture_webcam/{cam1_encoded}"
url_web_cam2 = f"http://127.0.0.1:6001/capture_webcam/{cam2_encoded}"
url_web_cam3 = f"http://127.0.0.1:6001/capture_webcam/{cam3_encoded}"

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

MODELOS_CACHE = {}

def get_modelo(camera, programa, tipo_neural):
    key = (camera, programa, tipo_neural)

    if key in MODELOS_CACHE:
        return MODELOS_CACHE[key]
    
    if tipo_neural == "resnet":
      
        path = f"dataset/cam{camera}_prog{programa}/modelo_resnet34.pth"
        model, device = treinar_modelo_new.carregar_modelo(path)
    elif tipo_neural == "resnet50":
        path = f"dataset/cam{camera}_prog{programa}/modelo_resnet50.pth"
        model, device = treinar_modelo_new.carregar_modelo_resnet50(path)
  
    elif tipo_neural == "mobilenetv2":
        path = f"dataset/cam{camera}_prog{programa}/modelo_mobilenetv2.pth"
        model, device = treinar_modelo_new.carregar_modelo_mobilenet(path)
    elif tipo_neural == "yolo":
        model = YOLO(f"runs/detect/treinos/cam{camera}_prog{programa}/weights/best.pt").to("cuda")
        device = "cuda" if torch.cuda.is_available() else "cpu"

    MODELOS_CACHE[key] = (model, device)
    return model, device



CONFIG_CACHE_CAMERAS = {}
CONFIG_CACHE = {}
def load_config_v2(camera, programa):
    cfg_key = (camera, programa)
    cfg_file = f'config_{camera}_{programa}.ini'

    try:
        mtime = os.path.getmtime(cfg_file)
    except FileNotFoundError:
        raise RuntimeError(f"Arquivo {cfg_file} não encontrado")

    cache_item = CONFIG_CACHE.get(cfg_key)

    # 🔐 proteção total
    if (
        cache_item is None
        or not isinstance(cache_item, dict)
        or cache_item.get('mtime') != mtime
    ):
        cfg = configparser.ConfigParser()
        cfg.read(cfg_file)

        CONFIG_CACHE[cfg_key] = {
            'cfg': cfg,
            'mtime': mtime
        }
        # print("Config recarregado")

    return CONFIG_CACHE[cfg_key]['cfg']

@app.route('/executar', methods=['POST'])
def processResult():
    try:
        
        global monitorando
        global camera_ociosa
        global ultima_atividade
        monitorando = True

       
        
        ultima_atividade = datetime.datetime.now()

        
        
        tick_start = cv2.getTickCount()
        # Obtém a data e hora atuais

        #programa = request.json.get('programa')  # Se estiver usando JSON no corpo
        data = request.json  # Receber JSON do corpo
        programa = data.get('programa')  # Acessar o valor do programa

        camera = data.get('camera')  # Acessar o valor do programa
        
         # Verificar se a câmera está em cache e atualizar se necessário
        cfg_key_cameras = (camera, programa)
        if cfg_key_cameras not in CONFIG_CACHE_CAMERAS:
            config_camera = configparser.ConfigParser()
            config_camera.read(f'cameras.ini')
            CONFIG_CACHE_CAMERAS[cfg_key_cameras] = config_camera
        else:
            config_camera = CONFIG_CACHE_CAMERAS[cfg_key_cameras ]
        
        serial_webcam1 = str(config_camera['CAMERAS']['cam_1'])
        serial_webcam2 = str(config_camera['CAMERAS']['cam_2'])
        serial_webcam3 = str(config_camera['CAMERAS']['cam_3'])  
        serial_basler_1 = str(config_camera['CAMERAS']['serial_number_basler_1'])
        serial_basler_2 = str(config_camera['CAMERAS']['serial_number_basler_2'])
        serial_basler_3 = str(config_camera['CAMERAS']['serial_number_basler_3'])
        
        # Processar a imagem como no código original
        if(camera == '1'):
            if(cam_tipo == "basler"):
              #url_cam = url_cam1
              serial_camera = serial_basler_1
            else:
              serial_camera = serial_webcam1
              #url_cam = url_web_cam1
            
        elif(camera == '2'):
            if(cam_tipo == "basler"):
              #url_cam = url_cam2
              serial_camera = serial_basler_2
            else:
              serial_camera = serial_webcam2
             #url_cam = url_web_cam2
        elif(camera == '3'):
            if(cam_tipo == "basler"):
              return jsonify({"error": f"Erro. Pendente parametro camera basler: exemplo: camera:1 e camera:2"})
              # serial_camera = serial_number_2
            else:
              serial_camera = serial_webcam3
        else:
             return jsonify({"error": f"Erro. Pendente parametro camera: exemplo: camera:1 e camera:2"})
        
        #if(cameras[serial_camera]["ociosa"] == True):
              #reativar_camera(serial_camera)
        #response = requests.get(url_cam)
        
           
        if(cam_tipo == "basler"):
           frame = get_latest_frame(serial_camera)
        elif(cam_tipo == "webcam"):
           frame = get_latest_webcam_frame(serial_camera)
        else:
           frame = get_png_frame()
        img = frame.copy() if frame is not None else None
       
        #atualizar_atividade_camera(serial_camera)
        # Ler o arquivo de configuração INI
        config = configparser.ConfigParser()
        config.read(f'config_{camera}_{programa}.ini')
        
        # Obter os parâmetros de ajuste de imagem
        contrast = float(config['ImageAdjustments']['contrast'])
        brightness = int(config['ImageAdjustments']['brightness'])
        gamma = float(config['ImageAdjustments']['gamma'])
        exposure = float(config['ImageAdjustments']['exposure'])
        
        rede_neural = config.getboolean('Ferramentas', 'rede_neural_artificial')
        if rede_neural:
            tipo_neural = str(config['Modelos']['tipo_rede_neural_artificial'])
            

            if tipo_neural == "resnet":
                modelo_resnet, device = get_modelo(camera, programa, "resnet")

            elif tipo_neural == "resnet50":
                modelo_resnet, device = get_modelo(camera, programa, "resnet50")

            elif tipo_neural == "mobilenetv2":
                file_modelo = f"dataset/cam{camera}_prog{programa}/modelo_mobilenetv2.pth"
                modelo_mobilenet, device = get_modelo(camera, programa, "mobilenetv2")

            elif tipo_neural == "yolo":
                modelo_yol , device = get_modelo(camera, programa, "yolo")
                classe_ok = str(config['Modelos']['peca_ok'])
                classe_nok = str(config['Modelos']['peca_nok'])

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
            template = cv2.imread(f'template_{camera}_{programa}.png', 0)

        # Carregar a imagem  de referência com base na camera e programa
        #if ponto_parafuso == True or textura  == True or posicao == True or cor == True or ponto_xy == True or pixel == True:
                #template_image = cv2.imread(f'ref_{camera}_{programa}_OK.png') 
                #template_image_NOK = cv2.imread(f'ref_{camera}_{programa}_NOK.png') 
               
                
               
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
        #image_ref = template_image  # Imagem para o teste de cor e parafuso 
        
           # Carregar a imagem referencia NOK
        #image_ref_nok = template_image_NOK

        # ja inicia arquivo de status
        config = configparser.ConfigParser()


        status_list = []  # Lista para armazenar os valores de status  

        # Carregar o arquivo status.ini se ele existir
        config.read(f'status_{camera}.ini')

        # Verificar se a seção "STATUS" já existe, caso contrário, criá-la
        if 'STATUS' not in config:
            config['STATUS'] = {}


        rois_data = []
        for i, (x, y, w, h,textura_tolerancia,pixel_tolerancia, threshold_cor,ponto_tolerancia) in enumerate(rois, start=1):
            circulo = False
            if(w==h):circulo = True
            roi_idex = i
            
            ###############################  RNA  #######################################annotated = results[0].plot()
            if rede_neural == True:
                coordenadas = (x,y,w,h)
                #status_rna = predizer_frame(camera,programa,roi_idex,img_gray,coordenadas)
                #gray = adjust_image_optimized_2(img, contrast, brightness, gamma, exposure)
                #status_rna = predizer_frame(camera,programa,roi_idex,gray,coordenadas)
                #status_rna = predizer_frame_new(camera,programa,roi_idex,img,coordenadas,0.45)
                #imagem_anotada = None
                
                if tipo_neural == "yolo":
                  status_rna  = monitorar_movimentos.analise_cabo_flex(modelo_yol,classe_ok,classe_nok,image,coordenadas,0.7)
                elif tipo_neural == "resnet" or tipo_neural == "resnet50":
                  status_rna  = treinar_modelo_new.classificar_roi(modelo_resnet,device,image,coordenadas)
                elif tipo_neural == "mobilenetv2":
                  status_rna  = treinar_modelo_new.classificar_roi_mobilenetv2(modelo_mobilenet,device,image,coordenadas)
                #if(imagem_anotada is not None):
                 #   image = imagem_anotada
            else:
                status_rna = True
            ##############################################################################

            #########################Análise de Cor ######################################
            if(cor == True):
                coordenadas = x,y,w,h
              
                #analise imagem atual
                status_cor = AnaliseCor.Cor_new(image,camera, programa,threshold_cor,coordenadas)
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
                status_texutura,media_textura = AnaliseTextura.new_textura_analise_OK(roi,camera,programa,gray,mask,coordenadas, textura_tolerancia, contrast, brightness, gamma, exposure)
                if(status_texutura==False):
                   status_texutura, media_textura = AnaliseTextura.new_textura_analise_NOK(roi,camera,programa,gray,mask,coordenadas,textura_tolerancia,contrast, brightness, gamma, exposure)
                   if(status_texutura==True):
                       status_texutura = False 
            else:
                status_texutura = True
            
            ######################## Análise de pixel (Quantify)###############################

            coordenadas = x,y,w,h
            if pixel:

                status_ok, media_roi = AnalisePixel.new_pixel_analise_OK(
                    roi, camera, programa, gray, mask, coordenadas,
                    pixel_tolerancia, contrast, brightness, gamma, exposure
                )

                if status_ok:
                    status_roi = True   # APROVADO

                else:
                    status_nok, media_roi = AnalisePixel.new_pixel_analise_NOK(
                        roi, camera, programa, gray, mask, coordenadas,
                        pixel_tolerancia, contrast, brightness, gamma, exposure
                    )

                    if status_nok:
                        status_roi = False  # REPROVADO
                    else:
                        status_roi = False  # INDEFINIDO = REPROVADO
            else:
                status_roi = True


            cX = 0
            cY = 0
            ################## Análise de ponto do parafuso - retorna a coordenadas x e y #######################################
            if ponto_parafuso == True:
                
                cX,cY,status_paratuso = AnaliseParafuso.new_ParafusoAnaliseOK(image,camera,programa,coordenadas,contrast,brightness,gamma,exposure,ponto_tolerancia)

                if(status_paratuso == False):
                    cX,cY,status_paratuso = AnaliseParafuso.new_ParafusoAnaliseNOK(image,camera,programa,coordenadas,contrast,brightness,gamma,exposure,ponto_tolerancia)
                    if(status_paratuso == True):
                        status_paratuso = False
                        

            else:
                status_paratuso = True
            
             ##################   Analise de ponto - retorna as coordenadas x e y #######################################
            if ponto_xy == True:

                #analise da imagem de referencia OK
                status_ponto_xy = True
                img_regiao_paraf,  cX, cY, min_val = centroPonto.centroDoponto(image,(x,y),w,contrast,brightness,gamma,exposure,ponto_tolerancia)
                if(cX != 0 and cY !=0):
                    status_ponto_xy = False
                       
            else:
                status_ponto_xy = True



            if( (status_rna and status_texutura and status_roi and status_cor and status_paratuso and status_posicao and status_ponto_xy) ):
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

            desenhar_roi(image, x, y, w, h, status, False, forma, True, f'ROI{i}')   

            status_list.append(status)  # Armazena o status para verificar depois
            
            if posicao:
                x_t, y_t = roi_template
                desenhar_roi(image, x_t, y_t, width_ref, height_ref, status, posicao, "retangulo",True, f'Localizador')
            rois_data.append({
                "roi_index": i,
                "media_textura": media_textura,
                "textura_status": status_texutura,
                "media_roi": media_roi,
                "pixel_status": status_roi,
                "cor_status":status_cor,
                "rna_status":status_rna,
                "rna_ativo":rede_neural,
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
            cv2.imwrite(f'imagem_status_{camera}.jpg', image)
        else:
            cv2.imwrite(f'imagem_status_{camera}.jpg', image)
        
        salvar_imagem_no_pendrive(camera,image)
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
        with open(f'status_{camera}.ini', 'w') as configfile:
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
    camera = data.get('camera')
    
    if not programa and camera:
        return jsonify({"error": "Número do programa ou camera não fornecido"}), 400
    
    base_filename = f"cam{camera}_ref_programa{programa}_OK"
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
    camera = data.get('camera')
    
    if not programa:
        return jsonify({"error": "Número do programa não fornecido"}), 400
    
    base_filename = f"cam{camera}_ref_programa{programa}_NOK"
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
        
        # exemplo de endpoint http://localhost:6001/processWithImage?programa=1&camera=1.

        programa = request.args.get('programa')  # Usando query string para GET
        camera = request.args.get('camera')  # Usando query string para GET
        # Ler o arquivo de configuração INI
        config = configparser.ConfigParser()
        config.read(f'config_{camera}_{programa}.ini')
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
        if (camera =='1'):
             if(cam_tipo == "basler"):
               response = requests.get(url_cam1)
             else:
               response = requests.get(url_web_cam1)
        elif(camera=='2'):
             if(cam_tipo == "basler"):
                 response = requests.get(url_cam2)
             else:
                 response = requests.get(url_web_cam2)
        elif(camera=='3'):
             if(cam_tipo == "basler"):
                 return jsonify({"error": f"Erro. Pendente parametro camera basler: exemplo: camera:1 e camera:2"})
             else:
                 response = requests.get(url_web_cam3)     
        else:
            return jsonify({"error": f"sem argumento camera"})
        
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

@app.route('/capture_png', methods=['GET'])
def capture_png():
    """Retorna frame a partir de um PNG"""

    frame = get_png_frame()

    if frame is not None:

        frame = cv2.resize(
            frame,
            (TARGET_WIDTH, TARGET_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        _, buffer = cv2.imencode('.jpg', frame)

        return Response(buffer.tobytes(), content_type='image/jpeg')

    else:
        return jsonify({"error": "Imagem PNG não encontrada"}), 500

@app.route('/capture/<serial_number>', methods=['GET'])
def capture_frame_new(serial_number):
    """Captura o frame mais recente da câmera especificada"""
    if (cam_tipo == "webcam"):
        frame = get_latest_webcam_frame(serial_number)
        if frame is not None:
                    frame = cv2.resize(
                        frame,
                        (TARGET_WIDTH, TARGET_HEIGHT),
                        interpolation=cv2.INTER_AREA
                    )
    elif(cam_tipo == "basler"):
        frame = get_latest_frame(serial_number)
    else:
        frame = get_png_frame()
    

    if frame is not None:
        _, buffer = cv2.imencode('.jpg', frame)
      
        return Response(buffer.tobytes(), content_type='image/jpeg')
    else:
        return jsonify({"error": "Nenhum frame disponível"}), 500
    

@app.route('/capture_webcam/<path:serial_number>', methods=['GET'])
def capture_frame_new_webcam(serial_number):
    """Captura o frame mais recente da câmera especificada"""
    if (cam_tipo == "webcam"):
       frame = get_latest_webcam_frame(serial_number)
       if frame is not None:
                    frame = cv2.resize(
                        frame,
                        (TARGET_WIDTH, TARGET_HEIGHT),
                        interpolation=cv2.INTER_AREA
                    )
        
    else:
        frame = get_latest_frame(serial_number)
    

    if frame is not None:
        _, buffer = cv2.imencode('.jpg', frame)
      
        return Response(buffer.tobytes(), content_type='image/jpeg')
    else:
        return jsonify({"error": "Nenhum frame disponível"}), 500




@app.route('/capture_reference', methods=['POST'])
def capture_frame_reference():
    data = request.json
    programa = data.get('programa')
    camera = data.get('camera')

    if camera == 1:
        if(cam_tipo == "basler"):
           url = url_cam1
        else:
           url = url_web_cam1 
    elif camera == 2:
        if(cam_tipo == "basler"):
           url = url_cam2
        else:
           url = url_web_cam2
    elif camera == 3:
        if(cam_tipo == "basler"):
           return jsonify({"error": f"Erro. Pendente parametro camera basler: exemplo: camera:1 e camera:2"})
        else:
           url = url_web_cam3
    else:
        return jsonify({"error": "Parâmetro 'camera' inválido. Use 1 ou 2."}), 400

    try:
        response = requests.get(url)
    except Exception as e:
        return jsonify({"error": f"Erro ao acessar URL da câmera: {str(e)}"}), 500

    if response.status_code == 200:
        try:
            nparr = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({"error": "Erro ao decodificar imagem recebida"}), 500

            base_filename = f"cam{camera}_ref_programa{programa}_OK"
            extension = ".png"
            filename = base_filename + extension
            filepath = os.path.join(STATIC_IMAGE_FOLDER, filename) #*
            counter = 1
            
            # Evita sobrescrever arquivos existentes
            while os.path.exists(filepath):
                filename = f"{base_filename}_{counter}{extension}"
                filepath = os.path.join(STATIC_IMAGE_FOLDER, filename) #*
                counter += 1

            #cv2.imwrite(filename, img)
            cv2.imwrite(filepath, img)
            return jsonify({"message": f"Frame capturado e salvo como '{filename}'"}), 200

        except Exception as e:
            return jsonify({"error": f"Erro ao processar imagem: {str(e)}"}), 500

    else:
        try:
            return jsonify({"error": f"Erro ao capturar frame: {response.json().get('error', 'desconhecido')}"}), 500
        except:
            return jsonify({"error": "Erro ao capturar frame e resposta inválida"}), 500
        
@app.route('/capture_reference_NOK', methods=['POST'])
def capture_frame_reference_NOK():
    data = request.json
    programa = data.get('programa')
    camera = data.get('camera')

    if camera == 1:
        if(cam_tipo == "basler"):
           url = url_cam1
        else:
           url = url_web_cam1 
    elif camera == 2:
        if(cam_tipo == "basler"):
           url = url_cam2
        else:
           url = url_web_cam2
    elif camera == 3:
        if(cam_tipo == "basler"):
           return jsonify({"error": f"Erro. Pendente parametro camera basler: exemplo: camera:1 e camera:2"})
        else:
           url = url_web_cam3

    try:
        response = requests.get(url)
    except Exception as e:
        return jsonify({"error": f"Erro ao acessar URL da câmera: {str(e)}"}), 500

    if response.status_code == 200:
        try:
            nparr = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({"error": "Erro ao decodificar imagem recebida"}), 500

            base_filename = f"cam{camera}_ref_programa{programa}_NOK"
            extension = ".png"
            filename = base_filename + extension
            filepath = os.path.join(STATIC_IMAGE_FOLDER, filename) #*
            counter = 1

            # Evita sobrescrever arquivos existentes
            while os.path.exists(filepath):
                filename = f"{base_filename}_{counter}{extension}"
                filepath = os.path.join(STATIC_IMAGE_FOLDER, filename) #*
                counter += 1

            cv2.imwrite(filepath, img)
            return jsonify({"message": f"Frame capturado e salvo como '{filename}'"}), 200

        except Exception as e:
            return jsonify({"error": f"Erro ao processar imagem: {str(e)}"}), 500

    else:
        try:
            return jsonify({"error": f"Erro ao capturar frame: {response.json().get('error', 'desconhecido')}"}), 500
        except:
            return jsonify({"error": "Erro ao capturar frame e resposta inválida"}), 500
        

STATIC_IMAGE_FOLDER = os.path.join(app.root_path, 'static', 'imagens')
os.makedirs(STATIC_IMAGE_FOLDER, exist_ok=True)

@app.route('/count_nok_files/<int:camera>/<int:program_number>', methods=['GET'])
def count_nok_files(camera,program_number):
    # Padrão esperado: "ref_programa{numero}_NOK" seguido de opcional "_X" e extensão .png
    pattern = re.compile(rf'^cam{camera}_ref_programa{program_number}_NOK(_\d+)?\.png$')
    
    # Lista todos os arquivos no diretório atual (ou ajuste o caminho)
    files = os.listdir(STATIC_IMAGE_FOLDER)  # Ou defina um diretório específico, ex: '/path/to/files'
    
    # Filtra os arquivos que correspondem ao padrão
    matching_files = [file for file in files if pattern.match(file)]
    
    # Retorna a contagem em JSON
    return jsonify({
        "program_number": program_number,
        "camera":camera,
        "count": len(matching_files),
        "matching_files": matching_files  # Opcional: listar os arquivos encontrados
    })

@app.route('/count_ok_files/<int:camera>/<int:program_number>', methods=['GET'])
def count_ok_files(camera,program_number):
    # Padrão esperado: "ref_programa{numero}_NOK" seguido de opcional "_X" e extensão .png
    pattern = re.compile(rf'^cam{camera}_ref_programa{program_number}_OK(_\d+)?\.png$')
    
    # Lista todos os arquivos no diretório atual (ou ajuste o caminho)
    files = os.listdir(STATIC_IMAGE_FOLDER)  # Ou defina um diretório específico, ex: '/path/to/files'
    
    # Filtra os arquivos que correspondem ao padrão
    matching_files = [file for file in files if pattern.match(file)]
    
    # Retorna a contagem em JSON
    return jsonify({
        "program_number": program_number,
        "camera":camera,
        "count": len(matching_files),
        "matching_files": matching_files  # Opcional: listar os arquivos encontrados
    })





@app.route('/imagens_ok', methods=['GET'])
def listar_imagens():
    camera = request.args.get('camera')
    programa = request.args.get('programa')

    if not camera or not programa:
        return "Parâmetros 'camera' e 'programa' são obrigatórios", 400

    padrao = f"cam{camera}_ref_programa{programa}_OK*.png"
    imagens = glob.glob(os.path.join(STATIC_IMAGE_FOLDER, padrao))
    imagens = [os.path.basename(img) for img in imagens]  # Apenas o nome dos arquivos

    # HTML simples para mostrar as imagens com botões de exclusão
    html = """
        <!DOCTYPE html>
        <html lang="pt-BR">
        <head>
            <meta charset="UTF-8">
            <title>Imagens Positivas</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #121212;
                    color: #f0f0f0;
                    padding: 20px;
                    margin: 0;
                }

                h1 {
                    text-align: center;
                    color: #ffffff;
                    margin-bottom: 30px;
                }
                
                .lista-imagens {
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                    max-width: 800px;
                    margin: 0 auto;
                }

                .item-imagem {
                    display: flex;
                    align-items: center;
                    background: #1e1e1e;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
                    padding: 15px;
                }

                .item-imagem img {
                    max-width: 300px;
                    height: auto;
                    border-radius: 5px;
                     border: 3px solid #2ecc71; /* verde vibrante */
                }

                .item-imagem .botao-excluir {
                    margin-left: auto;
                }

                .item-imagem button {
                    background-color: #e74c3c;
                    color: white;
                    border: none;
                    padding: 8px 14px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 14px;
                    transition: background 0.2s;
                }

                .item-imagem button:hover {
                    background-color: #c0392b;
                }

                .sem-imagens {
                    text-align: center;
                    font-style: italic;
                    color: #aaa;
                    margin-top: 40px;
                }
            </style>
        </head>
        <body>
            <h1>Imagens Positivas - Programa {{programa}} | Câmera {{camera}}</h1>
            <div style="text-align:center; margin-bottom: 20px;">
                <button onclick="excluirTodasImagens()" style="
                    background-color:#ff9800;
                    color:#000;
                    border:none;
                    padding:10px 18px;
                    border-radius:6px;
                    font-size:15px;
                    cursor:pointer;
                    font-weight:bold;
                ">
                    Excluir todas as imagens
                </button>
            </div>
            <div id="lista-imagens" class="lista-imagens">
                {% for imagem in imagens %}
                    <div class="item-imagem" data-filename="{{ imagem }}">
                        <div style="margin-right: 15px; font-weight: bold;">{{ loop.index }}.</div>
                        <img src="{{ url_for('static', filename='imagens/' + imagem) }}" alt="{{ imagem }}">
                        <div class="botao-excluir">
                            <button onclick="excluirImagem('{{ imagem }}')">Excluir</button>
                        </div>
                    </div>
                {% endfor %}

                {% if imagens|length == 0 %}
                    <p class="sem-imagens">Nenhuma imagem encontrada.</p>
                {% endif %}
            </div>

            <script>
                function excluirImagem(filename) {
                    if (!confirm("Deseja realmente excluir " + filename + "?")) return;

                    fetch("/apagar_imagem", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({ filename: filename })
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.message) {
                            // Remove o item da lista
                            const item = document.querySelector('[data-filename="' + filename + '"]');
                            if (item) {
                                item.remove();
                            }

                            // Se não houver mais imagens, mostra mensagem
                            if (document.querySelectorAll('.item-imagem').length === 0) {
                                document.getElementById('lista-imagens').innerHTML = "<p class='sem-imagens'>Nenhuma imagem restante.</p>";
                            }

                        } else if (data.error) {
                            alert("Erro: " + data.error);
                        }
                    })
                    .catch(err => {
                        alert("Erro ao excluir: " + err);
                    });
                }
                function excluirTodasImagens() {
                    if (!confirm("Deseja realmente excluir TODAS as imagens?")) return;

                    const imagens = document.querySelectorAll('.item-imagem');

                    if (imagens.length === 0) {
                        alert("Não há imagens para excluir.");
                        return;
                    }

                    let excluidas = 0;

                    imagens.forEach(item => {
                        const filename = item.getAttribute('data-filename');

                        fetch("/apagar_imagem", {
                            method: "POST",
                            headers: {
                                "Content-Type": "application/json"
                            },
                            body: JSON.stringify({ filename: filename })
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.message) {
                                item.remove();
                                excluidas++;

                                if (excluidas === imagens.length) {
                                    document.getElementById('lista-imagens').innerHTML =
                                        "<p class='sem-imagens'>Nenhuma imagem restante.</p>";
                                }
                            }
                        })
                        .catch(err => {
                            console.error("Erro ao excluir", filename, err);
                        });
                    });
                }
            </script>
        </body>
        </html>
        """


    return render_template_string(html, imagens=imagens, camera=camera, programa=programa)

@app.route('/imagens_nok', methods=['GET'])
def listar_imagens_nok():
    camera = request.args.get('camera')
    programa = request.args.get('programa')

    if not camera or not programa:
        return "Parâmetros 'camera' e 'programa' são obrigatórios", 400

    padrao = f"cam{camera}_ref_programa{programa}_NOK*.png"
    imagens = glob.glob(os.path.join(STATIC_IMAGE_FOLDER, padrao))
    imagens = [os.path.basename(img) for img in imagens]  # Apenas o nome dos arquivos

    # HTML simples para mostrar as imagens com botões de exclusão
    html = """
        <!DOCTYPE html>
        <html lang="pt-BR">
        <head>
            <meta charset="UTF-8">
            <title>Imagens Negativas</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #121212;
                    color: #f0f0f0;
                    padding: 20px;
                    margin: 0;
                }

                h1 {
                    text-align: center;
                    color: #ffffff;
                    margin-bottom: 30px;
                }

                .lista-imagens {
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                    max-width: 800px;
                    margin: 0 auto;
                }

                .item-imagem {
                    display: flex;
                    align-items: center;
                    background: #1e1e1e;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
                    padding: 15px;
                }

                .item-imagem img {
                    max-width: 300px;
                    height: auto;
                    border-radius: 5px;
                    border: 3px solid #e74c3c; 
                }

                .item-imagem .botao-excluir {
                    margin-left: auto;
                }

                .item-imagem button {
                    background-color: #e74c3c;
                    color: white;
                    border: none;
                    padding: 8px 14px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 14px;
                    transition: background 0.2s;
                }

                .item-imagem button:hover {
                    background-color: #c0392b;
                }

                .sem-imagens {
                    text-align: center;
                    font-style: italic;
                    color: #aaa;
                    margin-top: 40px;
                }
            </style>
        </head>
        <body>
            <h1>Imagens Negativas - Programa {{programa}} | Câmera {{camera}}</h1>
             <div style="text-align:center; margin-bottom: 20px;">
                <button onclick="excluirTodasImagens()" style="
                    background-color:#ff9800;
                    color:#000;
                    border:none;
                    padding:10px 18px;
                    border-radius:6px;
                    font-size:15px;
                    cursor:pointer;
                    font-weight:bold;
                ">
                    Excluir todas as imagens
                </button>
            </div>
            <div id="lista-imagens" class="lista-imagens">
                {% for imagem in imagens %}
                    <div class="item-imagem" data-filename="{{ imagem }}">
                        <div style="margin-right: 15px; font-weight: bold;">{{ loop.index }}.</div>
                        <img src="{{ url_for('static', filename='imagens/' + imagem) }}" alt="{{ imagem }}">
                        <div class="botao-excluir">
                            <button onclick="excluirImagem('{{ imagem }}')">Excluir</button>
                        </div>
                    </div>
                {% endfor %}

                {% if imagens|length == 0 %}
                    <p class="sem-imagens">Nenhuma imagem encontrada.</p>
                {% endif %}
            </div>

            <script>
                function excluirImagem(filename) {
                    if (!confirm("Deseja realmente excluir " + filename + "?")) return;

                    fetch("/apagar_imagem", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({ filename: filename })
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.message) {
                            // Remove o item da lista
                            const item = document.querySelector('[data-filename="' + filename + '"]');
                            if (item) {
                                item.remove();
                            }

                            // Se não houver mais imagens, mostra mensagem
                            if (document.querySelectorAll('.item-imagem').length === 0) {
                                document.getElementById('lista-imagens').innerHTML = "<p class='sem-imagens'>Nenhuma imagem restante.</p>";
                            }

                        } else if (data.error) {
                            alert("Erro: " + data.error);
                        }
                    })
                    .catch(err => {
                        alert("Erro ao excluir: " + err);
                    });
                }
                function excluirTodasImagens() {
                    if (!confirm("Deseja realmente excluir TODAS as imagens?")) return;

                    const imagens = document.querySelectorAll('.item-imagem');

                    if (imagens.length === 0) {
                        alert("Não há imagens para excluir.");
                        return;
                    }

                    let excluidas = 0;

                    imagens.forEach(item => {
                        const filename = item.getAttribute('data-filename');

                        fetch("/apagar_imagem", {
                            method: "POST",
                            headers: {
                                "Content-Type": "application/json"
                            },
                            body: JSON.stringify({ filename: filename })
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.message) {
                                item.remove();
                                excluidas++;

                                if (excluidas === imagens.length) {
                                    document.getElementById('lista-imagens').innerHTML =
                                        "<p class='sem-imagens'>Nenhuma imagem restante.</p>";
                                }
                            }
                        })
                        .catch(err => {
                            console.error("Erro ao excluir", filename, err);
                        });
                    });
                }
            </script>
        </body>
        </html>
        """




    return render_template_string(html, imagens=imagens, camera=camera, programa=programa)
@app.route('/apagar_imagem', methods=['POST'])
def apagar_imagem():
    filename = request.form.get('filename') or request.json.get('filename')
    if not filename:
        return jsonify({"error": "Nome do arquivo não fornecido"}), 400

    filepath = os.path.join(STATIC_IMAGE_FOLDER, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify({"message": f"Imagem '{filename}' apagada com sucesso"}), 200
    else:
        return jsonify({"error": "Arquivo não encontrado"}), 404




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
    
    #exemple de endpoint http://localhost:6001/config?programa=1&camera=1&section=ROI1.
    programa = request.args.get('programa')  # Usando query string para GET
    camera = request.args.get('camera')  # Usando query string para GET
    # Obtém o parâmetro de consulta "section" da URL
    section = request.args.get('section')

    PATH = f'config_{camera}_{programa}.ini'

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
    camera = data.get('camera')
    PATH = f'config_{camera}_{programa}.ini'

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
    camera = data.get('camera')  # Pegando o valor de 'programa' do JSON

    PATH = f'config_{camera}_{programa}.ini'
    
    config = load_config(PATH)
    # Atualiza os valores gerais se fornecidos, garantindo que sejam armazenados como "true"/"false"
    if 'nome_programa' in data:
        config['Ferramentas']['nome_programa'] = str(data['nome_programa']).lower()
    if 'cor' in data:
        config['Ferramentas']['cor'] = str(data['cor']).lower()
    if 'rede_neural_artificial' in data:
        config['Ferramentas']['rede_neural_artificial'] = str(data['rede_neural_artificial']).lower()
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
    
    #exemple de endpoint http://localhost:6001/ferramentas?programa=1&camera=1.
    programa = request.args.get('programa')  # Usando query string para GET
    camera = request.args.get('camera')  # Usando query string para GET

    PATH = f'config_{camera}_{programa}.ini'
    # Carrega as configurações atuais
    config = load_config(PATH)

    # Retorna os valores de cada configuração
    return jsonify({
        "nome_programa": config['Ferramentas'].get('nome_programa'), #retorna o nome do programa
        "cor": config['Ferramentas'].get('cor', 'false') == 'true',  # Converte de volta para booleano
        "rede_neural_artificial": config['Ferramentas'].get('rede_neural_artificial', 'false') == 'true',
        "textura": config['Ferramentas'].get('textura', 'false') == 'true',
        "pixel": config['Ferramentas'].get('pixel', 'false') == 'true',
        "ponto_parafuso": config['Ferramentas'].get('ponto_parafuso', 'false') == 'true',
        "ponto_xy": config['Ferramentas'].get('ponto_xy', 'false') == 'true',
        "posicao": config['Ferramentas'].get('posicao', 'false') == 'true',
        "n_rois": config.getint('Ferramentas', 'n_rois')
    })


# Endpoint POST para alterar os parâmetros de configuração
@app.route('/modelos', methods=['POST'])
def update_modelo_config():
    

    data = request.json

    programa = data.get('programa')  # Pegando o valor de 'programa' do JSON
    camera = data.get('camera')  # Pegando o valor de 'programa' do JSON

    PATH = f'config_{camera}_{programa}.ini'
    
    config = load_config(PATH)
    # Atualiza os valores gerais se fornecidos, garantindo que sejam armazenados como "true"/"false"
    if 'tipo_rede_neural_artificial' in data:
        config['Modelos']['tipo_rede_neural_artificial'] = str(data['tipo_rede_neural_artificial']).lower()

    # Salva as alterações no arquivo INI
    save_config(PATH,config)
    return jsonify({"message": "Configuração atualizada com sucesso!"})

@app.route('/modelos', methods=['GET'])
def get_modelo_config():
    
    #exemple de endpoint http://localhost:6001/modelos?programa=1&camera=1.
    programa = request.args.get('programa')  # Usando query string para GET
    camera = request.args.get('camera')  # Usando query string para GET

    PATH = f'config_{camera}_{programa}.ini'
    # Carrega as configurações atuais
    config = load_config(PATH)

    # Retorna os valores de cada configuração
    return jsonify({
        "tipo_rede_neural_artificial": config['Modelos'].get('tipo_rede_neural_artificial')
    })

# Endpoint POST para acessar os parâmetros de configuração
@app.route('/get_cameras', methods=['GET'])
def get_cameras_config():
    
   

    PATH = f'cameras.ini'
    # Carrega as configurações atuais
    config = load_config(PATH)

    # Retorna os valores de cada configuração
    return jsonify({
        "cam_1": config['CAMERAS'].get('cam_1'),
        "cam_2": config['CAMERAS'].get('cam_2'),
        "cam_3": config['CAMERAS'].get('cam_3'),
        "tipo": config['CAMERAS'].get('tipo'),
        "cam_basler_1": config['CAMERAS'].get('serial_number_basler_1'),
        "cam_basler_2": config['CAMERAS'].get('serial_number_basler_2'),
        "cam_basler_3": config['CAMERAS'].get('serial_number_basler_3')
    })



# Endpoint POST para alterar os parâmetros de configuração
@app.route('/coord_ref', methods=['POST'])
def update_coordRef_config():
    
    data = request.json

    programa = data.get('programa')  # Pegando o valor de 'programa' do JSON
    camera = data.get('camera')  # Pegando o valor de 'camera' do JSON
    roi = data.get('roi') 
    PATH = f'config_{camera}_{programa}.ini'
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
    camera = data.get('camera')  # Acessa o valor do programa
    PATH= f'config_{camera}_{programa}.ini'
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
    if camera == '1':
        if(cam_tipo == "basler"):
           url = url_cam1
        else:
           url = url_web_cam1 
    elif camera == '2':
        if(cam_tipo == "basler"):
           url = url_cam2
        else:
           url = url_web_cam2
    elif camera == '3':
        if(cam_tipo == "basler"):
           return jsonify({"error": f"Erro. Pendente parametro camera basler: exemplo: camera:1 e camera:2"})
        else:
           url = url_web_cam3

    response = requests.get(url)
        #response = requests.capture_frame()
    if response.status_code == 200:
            nparr = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Realizar o recorte
    recorte = img[y:y+height, x:x+width]

    # Salvar o recorte como PNG
    cv2.imwrite(f"template_{camera}_{programa}.png", recorte)

    roi_regiao_template = []

    x_interes = int(config['Referencia']['x_interes'])
    y_interes = int(config['Referencia']['y_interes'])
    w_interes = int(config['Referencia']['width_interes'])
    h_interes = int(config['Referencia']['height_interes'])

    
    roi_regiao_template.append((x_interes, y_interes, w_interes, h_interes))

    #template = cv2.imread(f'template_{camera}_{programa}.png', 0)

    #frame = cv2.imread(f'cam{camera}_ref_programa{programa}.png', 0)
  
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
    camera = request.args.get('camera')  # Usando query string para GET

    PATH = f'config_{camera}_{programa}.ini'

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

@app.route('/video_feed/<serial_number>')
def video_feed_new(serial_number):
    """ Transmite vídeo ao vivo da câmera especificada """
    
    def generate_frames():
        while True:
            if (cam_tipo == "webcam"):
                frame = get_latest_webcam_frame(serial_number)
                if frame is not None:
                    frame = cv2.resize(
                        frame,
                        (TARGET_WIDTH, TARGET_HEIGHT),
                        interpolation=cv2.INTER_AREA
                    )
            elif (cam_tipo == "basler"):
                frame = get_latest_frame(serial_number)
            else:
                frame = get_png_frame()
            
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)  # ~30 FPS

    # Verifica se a câmera existe
    if serial_number not in cameras:
        return jsonify({"error": f"Câmera {serial_number} não encontrada"}), 404

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_webcam/<path:serial_number>')
def video_feed_new_webcam(serial_number):
    """ Transmite vídeo ao vivo da câmera especificada """

    

    for key in cameras_web:
        if key.endswith(serial_number.strip()):
            cam_key = key
           
            break

    

    if cam_key is None:
        return jsonify({"error": f"Câmera {serial_number} não encontrada"}), 404

    def generate_frames_wecam():
        while True:
            if cam_tipo == "webcam":
                frame = get_latest_webcam_frame(cam_key)
                if frame is not None:
                    frame = cv2.resize(
                        frame,
                        (TARGET_WIDTH, TARGET_HEIGHT),
                        interpolation=cv2.INTER_AREA
                    )
                #frame = monitorar_movimentos.monitor_movimento(frame)
            elif cam_tipo == "basler":
                frame = get_latest_frame(cam_key)
            else:
                frame = get_png_frame()

            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)  # ~30 FPS

    return Response(generate_frames_wecam(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')






import time as t
import datetime
global turno_agora, turno_anterior, turno_atual
turno_atual = "iniciando"
turno_agora = None
turno_anterior = None
def obter_turno():
        
    global turno_atual
    while True: 
        try:   
            agora = datetime.datetime.now().time()

            # Definição dos horários de cada turno
            turno1_inicio = datetime.time(7, 0)
            turno1_fim    = datetime.time(16, 50)

            turno2_inicio = datetime.time(16, 50)
            turno2_fim    = datetime.time(22, 0)
            turno3_inicio = datetime.time(22, 0)
            turno3_fim    = datetime.time(7, 0)  

            # Turno 1
            if turno1_inicio <= agora < turno1_fim:
                turno_atual = "1º Turno"

            # Turno 2
            if turno2_inicio <= agora < turno2_fim:
                turno_atual = "2º Turno"
            

            # Turno 3 (começa 22:00 e vai até 07:00 do próximo dia)
            if agora >= turno3_inicio or agora < turno3_fim:
                turno_atual = "3º Turno"
            time.sleep(1)  # Verifica a cada 1 segundos
        except Exception as e:
            print(f"[ERRO] Erro ao obter turno: {e}")

# Inicia a thread de monitoramento de eventos
def monitoramento_turno():
    monitoring_thread = threading.Thread(target=obter_turno, daemon=True)
    monitoring_thread.start()

#turno_atual = obter_turno()
#print("Turno atual:", turno_atual)


# Controle de cooldown (em segundos)
COOLDOWN_ALERTA = 120  # 2 minutos

# Marca o último disparo de cada alerta
ultimo_alerta = {
    "cluster_carrinho": 0,
    "cluster_berco": 0,
    "berco_1_ocupado": 0,
    "berco_2_ocupado": 0,
    "sem_produto": 0,
    "operadores": 0,
    "ops8_aguardando_producao": 0,
    "novo_turno": 0
}

def pode_disparar(chave):
    """Verifica se já passou o tempo mínimo para novo alerta"""
    agora = time.time()
    if agora - ultimo_alerta[chave] >= COOLDOWN_ALERTA:
        ultimo_alerta[chave] = agora
        return True
    return False

import subprocess
def tocar_beep(audio_file="beep_trim.wav"):
    def play():
        subprocess.Popen(
            ["aplay", audio_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    threading.Thread(target=play, daemon=True).start()


facial_pause = False
facial_pause_lock = threading.Lock()
########################## cam 2 Reconhecimento facial #############################################3
ultimo_nome_processado = ""
matricula_atual = "000001"
processed_frames_facial = {}
processing_threads_facial = {}


# Rota para obter os limites de uma ROI
@app.route('/get_limits/<roi>', methods=['GET'])
def get_limits(roi):

    
    programa = request.args.get('programa')  # Usando query string para GET
    camera = request.args.get('camera')  # Usando query string para GET
    

    PATH = f'config_{camera}_{programa}.ini'

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
    camera = data.get('camera')  # Acessa o valor da camera
    PATH = f'config_{camera}_{programa}.ini'

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
    


@app.route('/status.json')
def status_json():
    #http://localhost:6001/status.json?camera=2

    camera = request.args.get('camera', type=int)

    if camera not in [1, 2]:
        return jsonify({'erro': 'Parâmetro camera inválido ou ausente (esperado: 1 ou 2)'}), 400
    
    ini_file = f'status_{camera}.ini'
    config = configparser.ConfigParser()
    config.read(ini_file)

    if 'STATUS' not in config:
        return jsonify({'erro': f'Seção STATUS não encontrada no arquivo ini{ini_file}'}), 500

    s = config['STATUS']
    data = {
        'camera':camera,
        'programa': int(s.get('programa', 0)),
        'nome_programa': s.get('nome_programa', 'desconhecido'),
        'n_rois': int(s.get('n_rois', 0)),
        'status_geral': s.getboolean('status_geral', fallback=False),
        'tempo_execucao': float(s.get('tempo_execucao', 0.0)),
        'imagem': f'imagem_status_{camera}.jpg'
    }

    # Adiciona os parciais (ROI)
    for i in range(1, 30):
        key = f'parcial_p{i}'
        data[key] = s.getboolean(key, fallback=False)

    return jsonify(data)

# Servir imagem diretamente se estiver na mesma pastaxxxxxxxx''x'
@app.route('/imagem_status.jpg')
def imagem_status():
      #http://localhost:6001/imagem_status.jpg?camera=2
    camera = request.args.get('camera', type=int)

    if camera not in [1, 2]:
        return jsonify({'erro': 'Parâmetro camera inválido ou ausente (esperado: 1 ou 2)'}), 400

    return send_file(f'imagem_status_{camera}.jpg', mimetype='image/jpeg')



import time  # Certifique-se que está importado no topo do arquivo

################################### secao reconhecimento facial ##########################################

import threading

def converter_tempo_para_decimal(tempo_str):
    try:
        # troca vírgula por ponto se vier no padrão PT-BR
        tempo_str = tempo_str.replace(",", ".")
        return round(float(tempo_str), 2)
    except Exception:
        return 0






@app.route('/formulario_cadastro')
def formulario_cadastro():
    return render_template("form.html")






# Iniciar o servidor Flask
if __name__ == '__main__':
    if(cam_tipo == "webcam"):
       
      
       initialize_all_webcams()  # Inicializa a câmera antes de iniciar o servidor
       
       
    elif(cam_tipo == "basler"):
       if (cam_tipo == "basler"):
           if(cam_basler_conexao == "usb"): 
                initialize_all_cameras_basler_usb() 
           else:
                initialize_all_cameras_basler_gige()
    else:
       print("Modo Simulacao: Nenhuma câmera real será inicializada. O servidor servirá imagens estáticas para teste.")      
    app.run(host='0.0.0.0', port=6001, use_reloader=False) 

