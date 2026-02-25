from flask import Flask, request, jsonify
import cv2
import numpy as np
import configparser

app = Flask(__name__)

# Função para calcular a diferença entre duas cores (em HSV)
def color_difference(hsv1, hsv2):
    return np.linalg.norm(np.array(hsv1) - np.array(hsv2))

@app.route('/executar', methods=['POST'])
def executar():
    try:
        programa = request.json.get('programa')  # Se estiver usando JSON no corpo
        # Carregar a imagem de referência com base no programa
        template_image = None
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
                return jsonify({"error": "Programa não encontrado."}), 404

        # Verificar se a imagem de referência foi carregada corretamente
        if template_image is None:
            return jsonify({"error": "Imagem de referência não encontrada."}), 400

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
                    return jsonify({"error": f"Chave ausente: {e} na seção {section_name}."}), 400
            else:
                return jsonify({"error": f"Seção {section_name} não encontrada no arquivo config.ini."}), 400

        # Calcular as cores médias das ROIs na imagem de referência
        reference_colors = []
        for (x, y, w, h) in rois:
            roi_ref = template_image[y:y+h, x:x+w]
            hsv_roi_ref = cv2.cvtColor(roi_ref, cv2.COLOR_BGR2HSV)
            mean_color_ref = cv2.mean(hsv_roi_ref)[:3]
            reference_colors.append(mean_color_ref)

        

        # Carregar a imagem da peça
        image = cv2.imread('captura.png')  # Substitua pelo caminho da sua imagem de teste

        if image is None:
            return jsonify({"error": "Erro ao processar a imagem da peça."}), 400

        # Definir um limiar de tolerância para a comparação
        threshold = 30  # Tolerância para diferença de cor

        # Analisar cada ROI na imagem da peça
        resultados = []
        for i, (x, y, w, h) in enumerate(rois):
            # Recortar a ROI da imagem
            status = False
            roi = image[y:y+h, x:x+w]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mean_color = cv2.mean(hsv_roi)[:3]
            
            # Comparar com a cor de referência correspondente
            difference = color_difference(reference_colors[i], mean_color)
            
            # Verificar se a cor está dentro do padrão
            if difference < threshold:
                status =True
                resultados.append({"ROI": i+1, "status": status})
            else:
                status = False
                resultados.append({"ROI": i+1, "status": status})

        return jsonify({"programa": programa, "resultados": resultados}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
