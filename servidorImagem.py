from flask import Flask, Response, jsonify
import cv2
import threading
from flask_cors import CORS

app = Flask(__name__)

# Permite acesso de qualquer origem
CORS(app)

# Variável global para a câmera
camera = None
lock = threading.Lock()  # Para evitar concorrência nos frames

def initialize_camera():
    """
    Inicializa a câmera antes de qualquer requisição.
    """
    global camera
    camera = cv2.VideoCapture(0)  # Abrir a câmera (use o índice correto)
    threading.Thread(target=camera_loop, daemon=True).start()

def camera_loop():
    """
    Mantém a câmera ativa.
    """
    global camera
    while True:
        with lock:
            if camera and camera.isOpened():
                ret, frame = camera.read()
                if not ret:
                    break

@app.route('/capture', methods=['GET'])
def capture_frame():
    """
    Captura um frame da câmera.
    """
    global camera
    with lock:
        if camera and camera.isOpened():
            ret, frame = camera.read()
            if not ret:
                return jsonify({"error": "Falha ao capturar frame"}), 500
            
            # Codifica o frame como JPEG para enviar como resposta
            _, buffer = cv2.imencode('.jpg', frame)
            response = Response(buffer.tobytes(), content_type='image/jpeg')
            return response
        else:
            return jsonify({"error": "Câmera não está disponível"}), 500

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

if __name__ == '__main__':
    initialize_camera()  # Inicializa a câmera antes de iniciar o servidor
    app.run(host='0.0.0.0', port=5001, debug=False)

