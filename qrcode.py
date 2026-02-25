import cv2

def decode_qr_code(image, roi=None):
    # Se uma ROI for fornecida, cortar a imagem para essa região
    if roi is not None:
        x, y, w, h = roi
        image = image[y:y+h, x:x+w]

    # Inicializar o detector de QR Code
    detector = cv2.QRCodeDetector()

    # Detectar e decodificar o QR Code
    data, bbox, _ = detector.detectAndDecode(image)

    # Verificar se um QR Code foi encontrado
    if bbox is not None:
        print(f"QR Code detectado: {data}")
        # Desenhar o contorno do QR Code na imagem
        for i in range(len(bbox[0])):
            pt1 = tuple(bbox[0][i].astype(int))
            pt2 = tuple(bbox[0][(i+1) % len(bbox[0])].astype(int))
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        #cv2.imshow("QR Code", image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    else:
        print("Nenhum QR Code encontrado.")
    return data, image

# Exemplo de uso
# image = cv2.imread('caminho/para/sua/imagem.jpg')
# roi = (x, y, largura, altura)
# data, image_with_qr = decode_qr_code(image, roi)