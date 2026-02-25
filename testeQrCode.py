import cv2



# Carregar a imagem
image = cv2.imread("qrcode2.png")

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
    cv2.imshow("QR Code", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Nenhum QR Code encontrado.")
