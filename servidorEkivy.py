from flask import Flask, request, Response, jsonify, send_file,stream_with_context
import configparser
import cv2
import threading
import os
from flask_cors import CORS
import requests
import json
import numpy as np
import time

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Rectangle
from kivy.config import Config
from threading import Thread

app = Flask(__name__)
CORS(app)

# Ativar a tela cheia - Kivy

###########################################################################  REGIAO DE TELA #################################################################################

Config.set('graphics', 'fullscreen', 'auto')
class StatusGeral(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (220, 150)  # Definir largura e altura fixas
        self.pos = (10, 900)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 100  # Tamanho da fonte
        self.bold = True
        self.halign = "center"  # Centraliza horizontalmente o texto
        self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        #self.text = "Conectando ao servidor..."
        self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho

        # Desenhar o fundo e a borda
        with self.canvas.before:
            self.bg_color = Color(0, 0.5, 0, 1)  # Cor inicial do fundo (verde escuro)
            self.rect = Rectangle(pos=self.pos, size=self.size)  # Retângulo de fundo

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def set_status_color(self, status):
        """Atualiza a cor de fundo e o texto conforme o status."""
        if status == "OK":
            self.bg_color.rgba = (0, 1, 0, 1)  # Verde
            self.text = "OK"
        elif status == "NOT_OK":
            self.bg_color.rgba = (1, 0, 0, 1)  # Vermelho
            self.text = "NOK"
        else:
            self.bg_color.rgba = (0.5, 0.5, 0.5, 1)  # Cinza
            self.text = "Status: Desconhecido"

class StatusParcial1(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (80, 60)  # Definir largura e altura fixas
        self.pos = (10, 700)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 40  # Tamanho da fonte
        self.bold = True
        self.halign = "center"  # Centraliza horizontalmente o texto
        self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "OK"
        self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho

        # Desenhar o fundo e a borda
        with self.canvas.before:
            self.bg_color = Color(0, 0.5, 0, 1)  # Cor inicial do fundo (verde escuro)
            self.rect = Rectangle(pos=self.pos, size=self.size)  # Retângulo de fundo

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def set_status_color(self, status):
        """Atualiza a cor de fundo e o texto conforme o status."""
        if status == "OK":
            self.bg_color.rgba = (0, 1, 0, 1)  # Verde
            self.text = "OK"
        elif status == "NOT_OK":
            self.bg_color.rgba = (1, 0, 0, 1)  # Vermelho
            self.text = "NOK"
        else:
            self.bg_color.rgba = (0.5, 0.5, 0.5, 1)  # Cinza
            self.text = "Status: Desconhecido"

class LabelParcial1(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (200, 700)  # Canto superior esquerdo (ajuste conforme resolução)
        

        self.font_size = 30  # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "Ferramenta 1"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

class StatusParcial2(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (80, 60)  # Definir largura e altura fixas
        self.pos = (10, 600)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 40  # Tamanho da fonte
        self.bold = True
        self.halign = "center"  # Centraliza horizontalmente o texto
        self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "OK"
        self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho

        # Desenhar o fundo e a borda
        with self.canvas.before:
            self.bg_color = Color(0, 0.5, 0, 1)  # Cor inicial do fundo (verde escuro)
            self.rect = Rectangle(pos=self.pos, size=self.size)  # Retângulo de fundo

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def set_status_color(self, status):
        """Atualiza a cor de fundo e o texto conforme o status."""
        if status == "OK":
            self.bg_color.rgba = (0, 1, 0, 1)  # Verde
            self.text = "OK"
        elif status == "NOT_OK":
            self.bg_color.rgba = (1, 0, 0, 1)  # Vermelho
            self.text = "NOK"
        else:
            self.bg_color.rgba = (0.5, 0.5, 0.5, 1)  # Cinza
            self.text = "Status: Desconhecido"
class LabelParcial2(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (200, 600)  # Canto superior esquerdo (ajuste conforme resolução)
        

        self.font_size = 30  # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "Ferramenta 2"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

class StatusParcial3(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (80, 60)  # Definir largura e altura fixas
        self.pos = (10, 500)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 40  # Tamanho da fonte
        self.bold = True
        self.halign = "center"  # Centraliza horizontalmente o texto
        self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "OK"
        self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho

        # Desenhar o fundo e a borda
        with self.canvas.before:
            self.bg_color = Color(0, 0.5, 0, 1)  # Cor inicial do fundo (verde escuro)
            self.rect = Rectangle(pos=self.pos, size=self.size)  # Retângulo de fundo

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def set_status_color(self, status):
        """Atualiza a cor de fundo e o texto conforme o status."""
        if status == "OK":
            self.bg_color.rgba = (0, 1, 0, 1)  # Verde
            self.text = "OK"
        elif status == "NOT_OK":
            self.bg_color.rgba = (1, 0, 0, 1)  # Vermelho
            self.text = "NOK"
        else:
            self.bg_color.rgba = (0.5, 0.5, 0.5, 1)  # Cinza
            self.text = "Status: Desconhecido"

class LabelParcial3(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (200, 500)  # Canto superior esquerdo (ajuste conforme resolução)
        

        self.font_size = 30  # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "Ferramenta 3"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status
    
    

class LabelInicializacao(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar redimensionamento automático
        self.size = (300, 50)  # Definir tamanho fixo para a Label
        self.pos_hint = {'center_x': 0.5, 'center_y': 0.5}  # Centraliza horizontal e verticalmente

        self.font_size = 100  # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "Conectando ao servidor..."
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

        
        

class MonitorApp(App):
    def build(self):
        
        layout = FloatLayout()  # Permite posicionamento absolutose
        self.label_inicial = LabelInicializacao()
        self.status_geral = StatusGeral()
        self.status_p1 = StatusParcial1()
        self.label_p1 = LabelParcial1()

        self.status_p2 = StatusParcial2()
        self.label_p2 = LabelParcial2()

        self.status_p3 = StatusParcial3()
        self.label_p3 = LabelParcial3()

        layout.add_widget(self.label_inicial)
        layout.add_widget(self.status_geral)

        layout.add_widget(self.status_p1)
        layout.add_widget(self.label_p1)

        layout.add_widget(self.status_p2)
        layout.add_widget(self.label_p2)

        layout.add_widget(self.status_p3)
        layout.add_widget(self.label_p3)

        # Atualizar o status a cada 1 segundos
         #Clock.schedule_interval(self.update_status, 1)

        return layout

    def update_status(self, resultado_p1  = "NOK",resultado_p2  = "NOK",  resultado_p3  = "NOK", resultado_geral  = "NOK"):
        try:
           
            if(resultado_p1 =="OK"):
               self.status_p1.set_status_color("OK") 
            else:
               self.status_p1.set_status_color("NOT_OK") 
            
            if(resultado_p2 =="OK"):
               self.status_p2.set_status_color("OK") 
            else:
               self.status_p2.set_status_color("NOT_OK") 
            
            if(resultado_p3 =="OK"):
               self.status_p3.set_status_color("OK") 
            else:
               self.status_p3.set_status_color("NOT_OK") 

            if(resultado_geral =="OK"):
               self.status_geral.set_status_color("OK") 
            else:
               self.status_geral.set_status_color("NOT_OK") 

            
                
        except requests.exceptions.RequestException:
            # Tratamento para erros de conexão
            self.status_label.text = "Erro de conexão com o servidor"
            self.status_label.color = (1, 0, 0, 1)  # Vermelho
        except Exception as e:
            # Tratamento para outros erros
            self.status_label.text = f"Erro: {str(e)}"
            self.status_label.color = (1, 0, 0, 1)  # Vermelho

###############################################################################################################################################################################
# Variável global para a câmera
camera = None
lock = threading.Lock()  # Para evitar concorrência nos frames
last_frame = None
def initialize_camera():
    global camera
    camera = cv2.VideoCapture(0)
    #camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 15)
    threading.Thread(target=camera_loop, daemon=True).start()


def camera_loop():
    global camera, last_frame
    while True:
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
    
    # Aplicar correção de gama (tabela de lookup otimizada)
    gamma_correction = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype="uint8")
    return cv2.LUT(adjusted, gamma_correction)


url = "http://127.0.0.1:6001/capture"


@app.route('/executar', methods=['GET'])
def processResult():
    try:
        resultado_p1 = "NOK"
        resultado_p2 = "NOK"
        resultado_p3 = "NOK"
        resultado_geral = "NOK"
        # Ler o arquivo de configuração INI
        config = configparser.ConfigParser()
        config.read('config.ini')
        
        # Obter os parâmetros de ajuste de imagem
        contrast = float(config['ImageAdjustments']['contrast'])
        brightness = int(config['ImageAdjustments']['brightness'])
        gamma = float(config['ImageAdjustments']['gamma'])

        # Obter as coordenadas das ROIs
        rois = []
        for i in range(1, 4):  # Três ROIs
            section_name = f'ROI{i}'
            if section_name in config:
                x = int(config[section_name]['x'])
                y = int(config[section_name]['y'])
                w = int(config[section_name]['width'])
                h = int(config[section_name]['height'])
                texturelowlimit  = float(config[section_name]['texturelowlimit'])
                texturehightlimit = float(config[section_name]['texturehightlimit'])
                pixellowlimit = float(config[section_name]['pixellowlimit'])
                pixelhighlimit = float(config[section_name]['pixelhighlimit'])

                rois.append((x, y, w, h,texturelowlimit,texturehightlimit,pixellowlimit,pixelhighlimit))
        # Processar a imagem como no código original
        response = requests.get(url)
        #response = requests.capture_frame()
        if response.status_code == 200:
            nparr = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            return jsonify({"error": f"Erro ao capturar frame: {response.json()['error']}"})

        # Ajustar a imagem
        gray = adjust_image_optimized(img, contrast, brightness, gamma)

        # Converter imagem em escala de cinza para BGR
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        rois_data = []
        for i, (x, y, w, h,texturelowlimit,texturehightlimit,pixellowlimit,pixelhighlimit) in enumerate(rois, start=1):
            # Recortar a ROI
            roi = gray[y:y+h, x:x+w]
            status = False
            # Análise de textura
            dft = cv2.dft(np.float32(roi), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

            media_textura = float(np.mean(magnitude_spectrum))  # Converter para float
            media_roi = float(cv2.mean(roi)[0])  # Converter para float
            if( (texturelowlimit < media_textura < texturehightlimit) and (pixellowlimit < media_roi < pixelhighlimit) ):
                status = True
            media_textura = f"{media_textura:.2f}"
            media_roi = f"{media_roi:.2f}"
            
            match i:
                                case 1:
                                            # Configurar o texto e a cor com base no status
                                    if status:
                                        resultado_p1 = "OK"                               
                                    else:
                                        resultado_p1 = "NOT_OK"   
                                    

                                case 2:

                                    if status:
                                        resultado_p2 = "OK"                               
                                    else:
                                        resultado_p2 = "NOT_OK"   
                                    
                                
                                case 3:

                                    if status:
                                        resultado_p3 = "OK"                               
                                    else:
                                        resultado_p3 = "NOT_OK"   
                                   
                                case _:
                                    print("Caso não encontrado")
            
            rois_data.append({
                "roi_index": i,
                "media_textura": media_textura,
                "media_roi": media_roi,
                "status":status
            })

        if(resultado_p1 == "OK" and resultado_p2 == "OK" and resultado_p3 =="OK" ):
            resultado_geral = "OK"
        else:
            resultado_geral = "NOT_OK"   


        MonitorApp.update_status(resultado_p1, resultado_p2,resultado_p3, resultado_geral)             

        # Codificar a imagem em formato JPEG
        # _, img_encoded = cv2.imencode('.jpg', gray_bgr)
        #img_bytes = img_encoded.tobytes()

        return jsonify({
            "rois_data": rois_data
            
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/processWithImage', methods=['GET'])
def process_image():
    try:

        # Ler o arquivo de configuração INI
        config = configparser.ConfigParser()
        config.read('config.ini')

        # Obter os parâmetros de ajuste de imagem
        contrast = float(config['ImageAdjustments']['contrast'])
        brightness = int(config['ImageAdjustments']['brightness'])
        gamma = float(config['ImageAdjustments']['gamma'])

        # Obter as coordenadas das ROIs
        rois = []
        for i in range(1, 4):  # Três ROIs
            section_name = f'ROI{i}'
            if section_name in config:
                x = int(config[section_name]['x'])
                y = int(config[section_name]['y'])
                w = int(config[section_name]['width'])
                h = int(config[section_name]['height'])
                texturelowlimit  = int(config[section_name]['texturelowlimit'])
                texturehightlimit = int(config[section_name]['texturehightlimit'])
                pixellowlimit = int(config[section_name]['pixellowlimit'])
                pixelhighlimit = int(config[section_name]['pixelhighlimit'])

                rois.append((x, y, w, h,texturelowlimit,texturehightlimit,pixellowlimit,pixelhighlimit))
        # Processar a imagem como no código original
        response = requests.get(url)
        if response.status_code == 200:
            nparr = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            return jsonify({"error": f"Erro ao capturar frame: {response.json()['error']}"})

        # Ajustar a imagem
        gray = adjust_image_optimized(img, contrast, brightness, gamma)

        # Converter imagem em escala de cinza para BGR
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        rois_data = []
        for i, (x, y, w, h,texturelowlimit,texturehightlimit,pixellowlimit,pixelhighlimit) in enumerate(rois, start=1):
            # Recortar a ROI
            roi = gray[y:y+h, x:x+w]

            # Análise de textura
            dft = cv2.dft(np.float32(roi), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

            media_textura = float(np.mean(magnitude_spectrum))  # Converter para float
            media_roi = float(cv2.mean(roi)[0])  # Converter para float
           
            rois_data.append({
                "roi_index": i,
                "media_textura": media_textura,
                "media_roi": media_roi
            })

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
IMAGE_PATH = 'captura.png'

# Função para carregar o arquivo de configuração
def load_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    return config

# Função para salvar alterações no arquivo de configuração
def save_config(config):
    with open(CONFIG_PATH, 'w') as configfile:
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


# Endpoint GET para obter os parâmetros de configuração
@app.route('/config', methods=['GET'])
def get_config():
    # Carrega o arquivo de configuração
    config = load_config()
    
    # Obtém o parâmetro de consulta "section" da URL
    section = request.args.get('section')
    
    # Verifica se uma seção específica foi solicitada
    if section:
        if section and config.has_section(section):
            # Retorna os dados da seção solicitada
            response = {
                "contrast": config.getfloat('ImageAdjustments', 'contrast'),
                "brightness": config.getint('ImageAdjustments', 'brightness'),
                "gamma": config.getfloat('ImageAdjustments', 'gamma'),
                "roi": {
                "x": config.getint(section, 'x'),
                "y": config.getint(section, 'y'),
                "width": config.getint(section, 'width'),
                "height": config.getint(section, 'height'),
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
            "rois": rois,
        }
    
    return jsonify(response)

# Endpoint POST para alterar os parâmetros de configuração
@app.route('/config', methods=['POST'])
def update_config():
    config = load_config()
    data = request.json
    
    # Atualiza os valores gerais se fornecidos
    if 'contrast' in data:
        config['ImageAdjustments']['contrast'] = str(data['contrast'])
    if 'brightness' in data:
        config['ImageAdjustments']['brightness'] = str(data['brightness'])
    if 'gamma' in data:
        config['ImageAdjustments']['gamma'] = str(data['gamma'])
    
    # Atualiza os valores da ROI específica se fornecidos
    if 'roi' in data and 'section' in data['roi']:
        section = data['roi']['section']  # Exemplo: "ROI1", "ROI2"
        if section in config:
            roi = data['roi']
            if 'x' in roi:
                config[section]['x'] = str(roi['x'])
            if 'y' in roi:
                config[section]['y'] = str(roi['y'])
            if 'width' in roi:
                config[section]['width'] = str(roi['width'])
            if 'height' in roi:
                config[section]['height'] = str(roi['height'])
        else:
            return jsonify({"error": f"Seção {section} não encontrada"}), 400
    
    # Salva as alterações no arquivo INI
    save_config(config)
    return jsonify({"message": "Configuração atualizada com sucesso!"})

# Endpoint GET para retornar a imagem
@app.route('/image', methods=['GET'])
def get_image():
    if os.path.exists(IMAGE_PATH):
        return send_file(IMAGE_PATH, mimetype='image/png')
    else:
        return jsonify({"error": "Imagem não encontrada"}), 404
    
# Rota para capturar imagens ao vivo
@app.route('/video_feed', methods=['GET'])
def video_feed():
    def generate_frames():
        while True:
            with lock:
                if last_frame is not None:
                    # Codifica o último frame como JPEG
                    _, buffer = cv2.imencode('.jpg', last_frame)
                    frame = buffer.tobytes()
                    # Adiciona as cabeçalhas do fluxo multipart
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)  # Reduz a taxa de atualização
    return Response(stream_with_context(generate_frames()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Rota para obter os limites de uma ROI
@app.route('/get_limits/<roi>', methods=['GET'])
def get_limits(roi):
    config = load_config()
    if roi not in config:
        return jsonify({"error": "ROI não encontrada"}), 404
    
    limits = {
        "texturelowlimit": int(config[roi]['texturelowlimit']),
        "texturehightlimit": int(config[roi]['texturehightlimit']),
        "pixellowlimit": int(config[roi]['pixellowlimit']),
        "pixelhighlimit": int(config[roi]['pixelhighlimit'])
    }
    return jsonify(limits)

# Rota para alterar os limites de uma ROI
@app.route('/set_limits/<roi>', methods=['POST'])
def set_limits(roi):
    config = load_config()
    if roi not in config:
        return jsonify({"error": "ROI não encontrada"}), 404

    # Obtém os limites enviados na requisição
    data = request.json
    required_keys = ["texturelowlimit", "texturehightlimit", "pixellowlimit", "pixelhighlimit"]
    if not all(key in data for key in required_keys):
        return jsonify({"error": "Faltando parâmetros necessários"}), 400
    
    # Atualiza os limites no arquivo
    config[roi]['texturelowlimit'] = str(data["texturelowlimit"])
    config[roi]['texturehightlimit'] = str(data["texturehightlimit"])
    config[roi]['pixellowlimit'] = str(data["pixellowlimit"])
    config[roi]['pixelhighlimit'] = str(data["pixelhighlimit"])

    # Escreve as alterações no arquivo
    save_config(config)

    return jsonify({"message": "Limites atualizados com sucesso!"}), 200


# Iniciar o servidor Flask
if __name__ == '__main__':
    initialize_camera()  # Inicializa a câmera antes de iniciar o servidor
    app.run(host='0.0.0.0', port=6001)

