from ultralytics import YOLO
import cv2
import time
import csv
import os
import json
import configparser


# === CONFIGURAÇÕES ===
CAM_INDEX = 2
MODEL_PATH = "yolov8n-pose.pt"
TEMPO_MINIMO = 2.0  # tempo mínimo de permanência (segundos)

# Lista de ROIs: [(x1, y1, x2, y2), ...]
ROIS = [
    #(100, 150, 500, 400),   # ROI 1
   # (600, 150, 1000, 400),  # ROI 2
]

# === INICIALIZAÇÃO ===
model = YOLO(MODEL_PATH)


#cv2.namedWindow("Monitor de Insercao - Multi ROI", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("Monitor de Insercao - Multi ROI", 1920, 1080)
# === AJUSTAR RESOLUÇÃO DA CÂMERA ===


# CSV de log
CSV_FILE = "registro_insercao.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["DataHora", "ROI", "Duracao", "Resultado"])

# === VARIÁVEIS DE ESTADO ===
estado = ["AGUARDANDO"] * len(ROIS)
inicio_insercao = [None] * len(ROIS)
cor_jig = [(255, 0, 0)] * len(ROIS)
prev_time = 0

# índices dos keypoints (COCO):
idx_pulso_esq, idx_pulso_dir = 9, 10
idx_cotovelo_esq, idx_cotovelo_dir = 7, 8


def load_config(PATH):
    config = configparser.ConfigParser()
    config.read(PATH)
    return config


def monitor_movimento(frame):
    # === LOOP PRINCIPAL ===
        global prev_time
        #ret, frame = cap.read()
        #if not ret:
           # break
        # Redimensiona a imagem para exibir maior (por exemplo, dobro do tamanho)
        #frame = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

       
        #PATH = f'config_{camera}_{programa}.ini'
        
        PATH = f'config_1_1.ini'

        # === CARREGAR CONFIGURAÇÃO ===
        config = load_config(PATH)
        n_rois = config.getint('Ferramentas', 'n_rois')

        ROIS = []
        for i in range(1, n_rois + 1):
            section_name = f'ROI{i}'
            if section_name in config:
                x = int(config[section_name]['x'])
                y = int(config[section_name]['y'])
                w = int(config[section_name]['width'])
                h = int(config[section_name]['height'])
                ROIS.append((x, y, w, h))

        # === SINCRONIZAR LISTAS DE ESTADO COM O NÚMERO DE ROIS ===
        n = len(ROIS)
        if 'estado' not in locals() or len(estado) != n:
            estado = ["AGUARDANDO"] * n
            inicio_insercao = [0] * n
            cor_jig = [(255, 0, 0)] * n

        # === PROCESSAMENTO DO FRAME ===
        results = model(frame, verbose=False)
        annotated_frame = frame.copy()
        h, w, _ = frame.shape

        pulsos = []

        for result in results:
            if result.keypoints is not None:
                for keypoints in result.keypoints.xy:
                    if len(keypoints) > max(idx_pulso_dir, idx_pulso_esq, idx_cotovelo_esq, idx_cotovelo_dir):
                        pulso_e = keypoints[idx_pulso_esq]
                        pulso_d = keypoints[idx_pulso_dir]
                        cotovelo_e = keypoints[idx_cotovelo_esq]
                        cotovelo_d = keypoints[idx_cotovelo_dir]

                        # desenhar punhos
                        cv2.circle(annotated_frame, (int(pulso_e[0]), int(pulso_e[1])), 5, (0, 255, 0), -1)
                        cv2.circle(annotated_frame, (int(pulso_d[0]), int(pulso_d[1])), 5, (0, 255, 0), -1)

                        # desenhar cotovelos
                        cv2.circle(annotated_frame, (int(cotovelo_e[0]), int(cotovelo_e[1])), 5, (255, 0, 0), -1)
                        cv2.circle(annotated_frame, (int(cotovelo_d[0]), int(cotovelo_d[1])), 5, (255, 0, 0), -1)

                        # Traçar linhas braço ↔ punho
                        cv2.line(annotated_frame, (int(cotovelo_e[0]), int(cotovelo_e[1])),
                                (int(pulso_e[0]), int(pulso_e[1])), (0, 255, 255), 2)
                        cv2.line(annotated_frame, (int(cotovelo_d[0]), int(cotovelo_d[1])),
                                (int(pulso_d[0]), int(pulso_d[1])), (0, 255, 255), 2)

                        pulsos.extend([pulso_e, pulso_d])


        # === AVALIAÇÃO DE CADA ROI ===
        for i, ROI in enumerate(ROIS):
            x1, y1, w, h = ROI
            x2, y2 = x1 + w, y1 + h

            dentro_roi = any(x1 < x < x2 and y1 < y < y2 for (x, y) in pulsos)

            # === LÓGICA DE ESTADO POR ROI ===
            if estado[i] == "AGUARDANDO" and dentro_roi:
                estado[i] = "INSERINDO"
                inicio_insercao[i] = time.time()
                cor_jig[i] = (0, 255, 255)

            elif estado[i] == "INSERINDO":
                if dentro_roi:
                    tempo_dentro = time.time() - inicio_insercao[i]
                    if tempo_dentro >= TEMPO_MINIMO:
                        estado[i] = "FINALIZADO"
                        cor_jig[i] = (0, 255, 0)
                        duracao = round(tempo_dentro, 2)
                        with open(CSV_FILE, "a", newline="") as f:
                            csv.writer(f).writerow([
                                time.strftime("%Y-%m-%d %H:%M:%S"), f"ROI {i + 1}", duracao, "OK ✅"
                            ])
                else:
                    duracao = round(time.time() - inicio_insercao[i], 2)
                    with open(CSV_FILE, "a", newline="") as f:
                        csv.writer(f).writerow([
                            time.strftime("%Y-%m-%d %H:%M:%S"), f"ROI {i + 1}", duracao, "INCOMPLETO ❌"
                        ])
                    estado[i] = "AGUARDANDO"
                    cor_jig[i] = (255, 0, 0)

            elif estado[i] == "FINALIZADO" and not dentro_roi:
                estado[i] = "AGUARDANDO"
                cor_jig[i] = (255, 0, 0)

            # === EXIBIÇÃO DE ROI ===
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), cor_jig[i], 2)
            cv2.putText(annotated_frame, f"ROI {i + 1}: {estado[i]}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 0, 128), 1)

            if estado[i] == "INSERINDO":
                tempo = time.time() - inicio_insercao[i]
                cv2.putText(annotated_frame, f"{tempo:.1f}s", (x1 + 10, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                # === CÁLCULO DE FPS ===
                current_time = time.time()
                if 'prev_time' not in globals():
                   prev_time = 0

                fps = 1 / (current_time - prev_time) if prev_time else 0
                prev_time = current_time

                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255,255), 1)

            
        return annotated_frame

        
