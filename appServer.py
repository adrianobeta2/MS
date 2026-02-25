from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
import configparser
import requests
import json


app = Flask(__name__)

def adjust_image_optimized(img, contrast=1.0, brightness=0, gamma=1.0):
    """
    Ajusta contraste, brilho e gama de uma imagem em um único passo.
    """
    # Converter para escala de cinza e ajustar contraste/brilho
    adjusted = cv2.convertScaleAbs(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), alpha=contrast, beta=brightness)
    
    # Aplicar correção de gama (tabela de lookup otimizada)
    gamma_correction = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype="uint8")
    return cv2.LUT(adjusted, gamma_correction)

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
        rois.append((x, y, w, h))

        
url = "http://127.0.0.1:6001/capture"



@app.route('/executar', methods=['GET'])
def processResult():
    try:
        
        

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
        for i, (x, y, w, h) in enumerate(rois, start=1):
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
        # _, img_encoded = cv2.imencode('.jpg', gray_bgr)
        #img_bytes = img_encoded.tobytes()

        return jsonify({
            "rois_data": rois_data
            
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

from flask import Response, jsonify

@app.route('/processWithImage', methods=['GET'])
def process_image():
    try:
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
        for i, (x, y, w, h) in enumerate(rois, start=1):
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



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
