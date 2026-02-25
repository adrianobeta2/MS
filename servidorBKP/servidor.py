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

app = Flask(__name__)
CORS(app)

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
            
            
            rois_data.append({
                "roi_index": i,
                "media_textura": media_textura,
                "media_roi": media_roi,
                "status":status
            })
       

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


# Iniciar o servidor Flask
if __name__ == '__main__':
    initialize_camera()  # Inicializa a câmera antes de iniciar o servidor
    app.run(host='0.0.0.0', port=6001)

