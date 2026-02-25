from ultralytics import YOLO
import cv2
import time
import math
import csv
from datetime import datetime
from collections import deque
import numpy as np

# --- Configuração ---
CSV_FILE = "movimentos.csv"
WINDOW = 5  # tamanho da janela para média móvel (suavização)

# Inicializa CSV
with open(CSV_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["movimento", "status", "inicio", "fim", "duracao_s"])

# --- Função para calcular ângulo entre três pontos ---
def calcular_angulo(a, b, c):
    ab = (a[0]-b[0], a[1]-b[1])
    cb = (c[0]-b[0], c[1]-b[1])
    dot = ab[0]*cb[0] + ab[1]*cb[1]
    norm_ab = math.hypot(*ab)
    norm_cb = math.hypot(*cb)
    if norm_ab*norm_cb == 0:
        return 0
    cos_theta = dot / (norm_ab * norm_cb)
    return math.degrees(math.acos(max(min(cos_theta,1),-1)))

# --- Inicializa YOLOv8 Pose ---
model = YOLO("yolov8n-pose.pt")

# Conexões do esqueleto (COCO)
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# --- Inicializa captura de vídeo ---
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    raise RuntimeError("❌ Não foi possível acessar a câmera.")

prev_time = 0

# Armazenar keypoints recentes para suavização
keypoints_buffer = deque(maxlen=WINDOW)

# Definir movimentos e seus status
movimentos = {
    "braço_direito": {"levantado": False, "inicio": 0},
    "braço_esquerdo": {"levantado": False, "inicio": 0},
    "tronco": {"inclinacao": False, "inicio": 0},
    "perna_direita": {"levantada": False, "inicio": 0},
    "perna_esquerda": {"levantada": False, "inicio": 0}
}

# Limiar dos movimentos (em porcentagem da altura)
THRESH_BRAÇO = 0.5    # 50% do comprimento ombro-quadril
THRESH_PERNA = 0.5     # 50% do comprimento quadril-joelho
THRESH_TRONCO = 0.2    # inclinação mínima do tronco

# --- Loop principal ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated_frame = frame.copy()

    for result in results:
        if result.keypoints is not None:
            for keypoints in result.keypoints.xy:
                # --- Suavizar keypoints com média móvel ---
                keypoints_buffer.append(keypoints)
                smoothed_keypoints = np.mean(np.array(keypoints_buffer), axis=0)

                # --- Desenhar esqueleto ---
                for (p1, p2) in POSE_CONNECTIONS:
                    x1, y1 = smoothed_keypoints[p1]
                    x2, y2 = smoothed_keypoints[p2]
                    cv2.line(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
                for x, y in smoothed_keypoints:
                    cv2.circle(annotated_frame, (int(x), int(y)), 4, (0, 255, 0), -1)

                # --- Normalizar posições ---
                altura_corpo = smoothed_keypoints[11][1] - smoothed_keypoints[5][1]  # quadril - ombro
                if altura_corpo == 0:
                    continue  # evitar divisão por zero

                # --- Braço direito ---
                punho_d = smoothed_keypoints[10]
                ombro_d = smoothed_keypoints[6]
                altura_relativa_d = (ombro_d[1] - punho_d[1]) / altura_corpo
                now = time.time()
                if altura_relativa_d > THRESH_BRAÇO and not movimentos["braço_direito"]["levantado"]:
                    movimentos["braço_direito"]["levantado"] = True
                    movimentos["braço_direito"]["inicio"] = now
                elif altura_relativa_d <= THRESH_BRAÇO and movimentos["braço_direito"]["levantado"]:
                    movimentos["braço_direito"]["levantado"] = False
                    duracao = now - movimentos["braço_direito"]["inicio"]
                    with open(CSV_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["braço_direito","levantado",
                                         datetime.fromtimestamp(movimentos["braço_direito"]["inicio"]).strftime("%H:%M:%S"),
                                         datetime.fromtimestamp(now).strftime("%H:%M:%S"),
                                         f"{duracao:.2f}"])

                # --- Braço esquerdo ---
                punho_e = smoothed_keypoints[9]
                ombro_e = smoothed_keypoints[5]
                altura_relativa_e = (ombro_e[1] - punho_e[1]) / altura_corpo
                if altura_relativa_e > THRESH_BRAÇO and not movimentos["braço_esquerdo"]["levantado"]:
                    movimentos["braço_esquerdo"]["levantado"] = True
                    movimentos["braço_esquerdo"]["inicio"] = now
                elif altura_relativa_e <= THRESH_BRAÇO and movimentos["braço_esquerdo"]["levantado"]:
                    movimentos["braço_esquerdo"]["levantado"] = False
                    duracao = now - movimentos["braço_esquerdo"]["inicio"]
                    with open(CSV_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["braço_esquerdo","levantado",
                                         datetime.fromtimestamp(movimentos["braço_esquerdo"]["inicio"]).strftime("%H:%M:%S"),
                                         datetime.fromtimestamp(now).strftime("%H:%M:%S"),
                                         f"{duracao:.2f}"])

                # --- Tronco ---
                meio_ombro = ( (ombro_e[0]+ombro_d[0])/2, (ombro_e[1]+ombro_d[1])/2 )
                meio_quadril = ( (smoothed_keypoints[11][0]+smoothed_keypoints[12][0])/2,
                                 (smoothed_keypoints[11][1]+smoothed_keypoints[12][1])/2 )
                inclinacao_relativa = abs(meio_ombro[1] - meio_quadril[1]) / altura_corpo
                if inclinacao_relativa < THRESH_TRONCO and not movimentos["tronco"]["inclinacao"]:
                    movimentos["tronco"]["inclinacao"] = True
                    movimentos["tronco"]["inicio"] = now
                elif inclinacao_relativa >= THRESH_TRONCO and movimentos["tronco"]["inclinacao"]:
                    movimentos["tronco"]["inclinacao"] = False
                    duracao = now - movimentos["tronco"]["inicio"]
                    with open(CSV_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["tronco","inclinacao",
                                         datetime.fromtimestamp(movimentos["tronco"]["inicio"]).strftime("%H:%M:%S"),
                                         datetime.fromtimestamp(now).strftime("%H:%M:%S"),
                                         f"{duracao:.2f}"])

                # --- Perna direita ---
                quadril_d = smoothed_keypoints[12]
                joelho_d = smoothed_keypoints[14]
                tornozelo_d = smoothed_keypoints[16]
                altura_perna_d = (quadril_d[1] - tornozelo_d[1]) / altura_corpo
                if altura_perna_d > THRESH_PERNA and not movimentos["perna_direita"]["levantada"]:
                    movimentos["perna_direita"]["levantada"] = True
                    movimentos["perna_direita"]["inicio"] = now
                elif altura_perna_d <= THRESH_PERNA and movimentos["perna_direita"]["levantada"]:
                    movimentos["perna_direita"]["levantada"] = False
                    duracao = now - movimentos["perna_direita"]["inicio"]
                    with open(CSV_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["perna_direita","levantada",
                                         datetime.fromtimestamp(movimentos["perna_direita"]["inicio"]).strftime("%H:%M:%S"),
                                         datetime.fromtimestamp(now).strftime("%H:%M:%S"),
                                         f"{duracao:.2f}"])

                # --- Perna esquerda ---
                quadril_e = smoothed_keypoints[11]
                joelho_e = smoothed_keypoints[13]
                tornozelo_e = smoothed_keypoints[15]
                altura_perna_e = (quadril_e[1] - tornozelo_e[1]) / altura_corpo
                if altura_perna_e > THRESH_PERNA and not movimentos["perna_esquerda"]["levantada"]:
                    movimentos["perna_esquerda"]["levantada"] = True
                    movimentos["perna_esquerda"]["inicio"] = now
                elif altura_perna_e <= THRESH_PERNA and movimentos["perna_esquerda"]["levantada"]:
                    movimentos["perna_esquerda"]["levantada"] = False
                    duracao = now - movimentos["perna_esquerda"]["inicio"]
                    with open(CSV_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["perna_esquerda","levantada",
                                         datetime.fromtimestamp(movimentos["perna_esquerda"]["inicio"]).strftime("%H:%M:%S"),
                                         datetime.fromtimestamp(now).strftime("%H:%M:%S"),
                                         f"{duracao:.2f}"])

                # --- Exibir status na tela ---
                cv2.putText(annotated_frame, f"Braco D: {'Levantado' if movimentos['braço_direito']['levantado'] else 'Baixo'}", (10,50), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                cv2.putText(annotated_frame, f"Braco E: {'Levantado' if movimentos['braço_esquerdo']['levantado'] else 'Baixo'}", (10,80), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                cv2.putText(annotated_frame, f"Tronco: {'Inclinado' if movimentos['tronco']['inclinacao'] else 'Normal'}", (10,110), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                cv2.putText(annotated_frame, f"Perna D: {'Levantada' if movimentos['perna_direita']['levantada'] else 'Baixo'}", (10,140), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                cv2.putText(annotated_frame, f"Perna E: {'Levantada' if movimentos['perna_esquerda']['levantada'] else 'Baixo'}", (10,170), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    # --- FPS ---
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    # --- Mostrar frame ---
    cv2.imshow("Monitor de Movimento - YOLO Pose", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
