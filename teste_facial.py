import cv2
from facial import reconhecer_api   # ajuste para o nome do seu script

VIDEO_PATH = "saida2.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo.")
        break

    # Processa o frame como se fosse da câmera
    annotated , ultimo_nome= reconhecer_api(frame)
    annotated, ultimo_nome = reconhecer_api(frame)

    # Escreve o nome no frame
    cv2.putText(annotated, 
                ultimo_nome,                 # texto
                (10, 30),                    # posição (x, y)
                cv2.FONT_HERSHEY_SIMPLEX,    # fonte
                1.0,                         # escala do texto
                (0, 255, 0),                 # cor (verde)
                2,                            # espessura
                cv2.LINE_AA)



    
    # Mostra o resultado
    cv2.imshow("Teste MP4", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
