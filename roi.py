import cv2

# Carrega a imagem
img = cv2.imread("imagem.png")

if img is None:
    raise ValueError("Não foi possível carregar a imagem")

# Seleciona ROI (clicar e arrastar)
roi = cv2.selectROI(
    "Selecione a ROI (ENTER confirma | ESC cancela)",
    img,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyAllWindows()

x, y, w, h = roi

print(f"ROI selecionada:")
print(f"x = {x}, y = {y}, w = {w}, h = {h}")
