import cv2
import time
import csv
import os
import configparser
from ultralytics import YOLO

# === CONFIGURAÇÕES GERAIS ===
MODEL_PATH = "best v5.pt"
CLASSE_ALVO = "cluster"
TEMPO_MINIMO = 2.0

CSV_FILE = "registro_insercao.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["DataHora", "ROI", "Duracao", "Resultado"])

# === CONFIGURAÇÃO DE ROIs ===
PATH = "config_1_1.ini"
config = configparser.ConfigParser()
config.read(PATH)
n_rois = config.getint('Ferramentas', 'n_rois')

ROIS = []
for i in range(1, n_rois + 1):
    section = f'ROI{i}'
    if section in config:
        x = int(config[section]['x'])
        y = int(config[section]['y'])
        w = int(config[section]['width'])
        h = int(config[section]['height'])
        ROIS.append((x, y, w, h))

# === ESTADO GLOBAL ===
contador_ids = set()
contador_total = 0
prev_time = 0

# === MODELO YOLO ===
model = YOLO(MODEL_PATH)


# === FUNÇÃO PRINCIPAL ===
def monitor_movimento(frame):
    """
    Detecta e rastreia objetos no frame, destacando e contando os da classe-alvo dentro da ROI.
    Retorna o frame anotado (annotated_frame).
    """
    global contador_ids, contador_total, prev_time

    # Executa a detecção no frame atual
    results = model.track(frame, persist=True, verbose=False)
    annotated_frame = frame.copy()

    # Se não houver resultado válido
    if not results or not results[0].boxes:
        return annotated_frame

    result = results[0]
    boxes = result.boxes
    names = model.names

    # --- Contador e desenho de objetos ---
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        track_id = int(box.id[0]) if box.id is not None else None

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # --- Cor personalizada ---
        color_box = (255, 255, 0) if cls_name == CLASSE_ALVO else (255, 0, 255)  # Ciano / Magenta
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_box, 2)
        cv2.circle(annotated_frame, (cx, cy), 4, color_box, -1)
        if track_id is not None:
            cv2.putText(annotated_frame, f"{cls_name} ID {track_id}", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

        # --- Contador dentro das ROIs ---
        for i, ROI in enumerate(ROIS):
            x_r, y_r, w_r, h_r = ROI
            dentro_roi = (x_r < cx < x_r + w_r) and (y_r < cy < y_r + h_r)

            if cls_name == CLASSE_ALVO and track_id is not None and dentro_roi:
                if track_id not in contador_ids:
                    contador_ids.add(track_id)
                    contador_total += 1
                    print(f"🖐️ Nova {CLASSE_ALVO} detectada (ID {track_id}) dentro da ROI {i+1}")

    # --- Desenhar ROIs de busca semi-transparentes ---
    overlay = annotated_frame.copy()
    for i, ROI in enumerate(ROIS):
        x, y, w, h = ROI
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), -1)
    annotated_frame = cv2.addWeighted(overlay, 0.25, annotated_frame, 0.75, 0)

    # --- Bordas e textos das ROIs ---
    for i, ROI in enumerate(ROIS):
        x, y, w, h = ROI
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"ROI {i+1}", (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- FPS ---
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # --- Total dentro da ROI ---
    cv2.putText(annotated_frame, f"Total {CLASSE_ALVO} na ROI: {contador_total}",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    return annotated_frame


# === EXEMPLO DE USO ===
if __name__ == "__main__":
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        success, frame = cap.read()
        if not success:
            break

        annotated = monitor_movimento(frame)
        cv2.imshow("Monitoramento - YOLOv8", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
