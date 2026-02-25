import cv2

# Abre a webcam
cap = cv2.VideoCapture("rtsp://admin:V@ssoura1331@192.168.1.64:554/Streaming/Channels/101")

# Verifica se a webcam abriu
if not cap.isOpened():
    print("Erro ao acessar a câmera")
    exit()

# Configurações do vídeo
largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30  # escolha o FPS desejado

# DEFINA o codec MP4
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec para MP4

# Cria o objeto de gravação
out = cv2.VideoWriter("saida7.mp4", fourcc, fps, (largura, altura))

print("Gravando... pressione Q para parar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Escreve o frame no arquivo
    out.write(frame)

    # Mostra na tela
    cv2.imshow("Gravando", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Vídeo salvo como saida.mp4")
