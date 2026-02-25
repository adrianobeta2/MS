import cv2

def capturar_imagem_com_estabilizacao(brilho=46, contraste=64, frames_estabilizacao=30, nome_arquivo="captura.png"):
    """
    Captura uma imagem após estabilizar a câmera.

    Parâmetros:
        brilho (int): Nível de brilho da câmera (padrão: 46).
        contraste (int): Nível de contraste da câmera (padrão: 64).
        frames_estabilizacao (int): Número de frames a processar antes da captura.
        nome_arquivo (str): Nome do arquivo onde a imagem será salva.
    """
    # Inicializa a captura da câmera (0 é o índice da câmera padrão)
    cap = cv2.VideoCapture(0)

    # Define propriedades da câmera
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brilho)
    cap.set(cv2.CAP_PROP_CONTRAST, contraste)

    # Loop para estabilizar a imagem
    for i in range(frames_estabilizacao):
        ret, _ = cap.read()
        if not ret:
            print("Falha ao estabilizar a câmera. Verifique a conexão.")
            cap.release()
            return

    # Captura o frame estabilizado
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar a imagem.")
        cap.release()
        return

    # Salva a imagem como PNG
    cv2.imwrite(nome_arquivo, frame)
    print(f"Imagem capturada e salva como '{nome_arquivo}'.")

    # Libera a câmera
    cap.release()

# Exemplo de uso
#capturar_imagem_com_estabilizacao(brilho=50, contraste=70, frames_estabilizacao=30, nome_arquivo="imagem_estabilizada.png")


    