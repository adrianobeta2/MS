import cv2

# Inicializa a captura da câmera (0 é o índice da câmera padrão)
cap = cv2.VideoCapture(0)

# Define propriedades da câmera
cap.set(cv2.CAP_PROP_BRIGHTNESS, 46)  # Ajusta o brilho
cap.set(cv2.CAP_PROP_CONTRAST, 64)    # Ajusta o contraste

print("Pressione 's' para capturar e salvar uma imagem em PNG.")
print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()  # Captura o frame da câmera
    if not ret:
        print("Falha ao acessar a câmera. Verifique a conexão.")
        break

    # Mostra o feed da câmera
    cv2.imshow("Camera", frame)

    # Aguarda a tecla pressionada
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # Se a tecla 's' for pressionada
        # Salva a imagem como PNG
        cv2.imwrite("captura.png", frame)
        #print("Imagem salva como 'capturaNOK.png'.")

    elif key == ord('q'):  # Se a tecla 'q' for pressionada
        print("Saindo...")
        break

# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()
