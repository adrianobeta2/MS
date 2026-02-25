from ultralytics import YOLO
import cv2

# Caminhos
MODEL_PATH = "cabo_flex_pcb_n.pt"
IMAGE_PATH = "ok.png"  # coloque uma imagem REAL do dataset

# Carrega modelo
model = YOLO(MODEL_PATH)

# Lê imagem
img = cv2.imread(IMAGE_PATH)
assert img is not None, "Erro ao carregar imagem"

# Inferência
results = model(img, conf=0.25, verbose=True)

# Debug visual + print
for r in results:
    if r.boxes is None:
        print("❌ Nenhuma box detectada")
        continue

    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Classe: {model.names[cls]} | Conf: {conf:.2f}")

# Mostrar imagem anotada
annotated = results[0].plot()
cv2.imshow("YOLO TESTE", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
