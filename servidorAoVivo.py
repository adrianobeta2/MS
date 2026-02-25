from flask import Flask, Response, stream_with_context
import cv2

app = Flask(__name__)
cap = cv2.VideoCapture(0)  # Captura da câmera

# Rota para capturar imagens ao vivo
@app.route('/video_feed', methods=['GET'])
def video_feed():
    def generate_frames():
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                # Codifica o quadro como JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                # Adiciona as cabeçalhas do fluxo multipart
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(stream_with_context(generate_frames()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Rota para capturar um único quadro
@app.route('/capture_frame', methods=['GET'])
def capture_frame():
    success, frame = cap.read()
    if not success:
        return {"error": "Could not capture frame"}, 500
    _, buffer = cv2.imencode('.jpg', frame)
    response = buffer.tobytes()
    return Response(response, content_type='image/jpeg')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
