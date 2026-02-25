import cv2

def capturarImg(brilho, contraste):
    # Inicializa a captura da câmera (0 é o índice da câmera padrão)
    cap = cv2.VideoCapture(0)

    # Define propriedades da câmera
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brilho)  # Ajusta o brilho
    cap.set(cv2.CAP_PROP_CONTRAST, contraste)    # Ajusta o contraste
    while True:
      ret, frame = cap.read()  # Captura o frame da câmera
      if not ret:
         print("Falha ao acessar a câmera. Verifique a conexão.")
         break

       # Mostra o feed da câmera
      cv2.imshow("Camera", frame)

        # Salva a imagem como PNG
      cv2.imwrite("captura.png", frame)
      print("Imagem salva!")
      break

    # Libera a câmera e fecha as janelas
    cap.release()
    cv2.destroyAllWindows()