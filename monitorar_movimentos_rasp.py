import cv2
import time
import csv
import os
import configparser
from ultralytics import YOLO
import threading


#from playsound import playsound

# === CONFIGURAÇÕES GERAIS ===
MODEL_PATH = "best_v6.pt"
#MODEL_PATH_POSE = "yolov8n-pose.pt"
MODEL_PATH_POSE = "operadores.pt"

CLASSE_ALVO = "cluster"
CLASSE_ALVO_2 = "maos_com_luva"
CLASSE_ALVO_3 = "operador"

TEMPO_REMOCAO_CLUSTER = 3.0
TIME_LIMIT_SEM_OBJETO = 100.0  # segundos
ULTRAPASSOU_LIMITE = False  
TOCAR_BEEP= False

CSV_FILE = "registro_insercao.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["DataHora", "ROI", "Duracao", "Resultado"])

# === CONFIGURAÇÃO DE ROIs ===
#PATH = "config_1_1.ini"
config = configparser.ConfigParser()

# === ESTADO GLOBAL ===
contador_ids = set()
contador_total = 0
contador_total_carrinho = 0
contador_total_pessoas = 0
cluster_no_berco = False
prev_time = 0
tempo_sem_objeto = 0.0
ultimo_tempo = time.time()

# === MODELO YOLO ===
model = YOLO(MODEL_PATH)
model_pose = YOLO(MODEL_PATH_POSE)
# controle do tempo de permanência de cluster por ROI
tempo_cluster_roi = []

#def tocar_beep():
   # threading.Thread(target=playsound, args=("beep_trim.mp3",), daemon=True).start()

# IDs que já tocaram beep por ROI
beep_ids_por_roi = []


def monitor_movimento(frame, camera, programa):
    """
    Recebe um frame já obtido externamente, realiza detecção + rastreamento,
    cronometra tempo sem objetos dentro da ROI e retorna annotated_frame.
    """
    
    global beep_ids_por_roi

    

    # tipo 1: conta apenas se cluster + mao_com_luva juntos
    # tipo 2: conta apenas se cluster sozinho
    global contador_ids, contador_total, prev_time, tempo_sem_objeto, ultimo_tempo

    # --- Ler ROIs do INI ---
    ROIS = []
    PATH = f"config_{camera}_{programa}.ini"
    config.read(PATH)
    n_rois = config.getint('Ferramentas', 'n_rois')
    for i in range(1, n_rois + 1):
        section = f'ROI{i}'
        if section in config:
            x = int(config[section]['x'])
            y = int(config[section]['y'])
            w = int(config[section]['width'])
            h = int(config[section]['height'])
            ROIS.append((x, y, w, h))
    global tempo_cluster_roi

    if len(tempo_cluster_roi) != len(ROIS):
        tempo_cluster_roi = [0.0] * len(ROIS)
    
    if len(beep_ids_por_roi) != len(ROIS):
        beep_ids_por_roi = [set() for _ in ROIS]

    # --- Detecção + rastreamento ---
    results = model.track(frame, persist=True, verbose=False)
    if not results:
        return frame

    res = results[0]
    annotated_frame = frame.copy()
    boxes = res.boxes
    names = model.names

    # Flag para saber se há algum objeto da classe-alvo dentro da ROI
    objeto_dentro_roi = False

   # --- Flags por ROI ---
    roi_tem_cluster = [False] * len(ROIS)
    roi_tem_mao = [False] * len(ROIS)
    track_ids_novos = [set() for _ in ROIS]

    # --- Processar todas as detecções ---
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        track_id = int(box.id[0]) if box.id is not None else None

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Desenho padrão (opcional)
        color_box = (255, 255, 0) if cls_name == CLASSE_ALVO else (255, 0, 255)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_box, 2)
        cv2.circle(annotated_frame, (cx, cy), 4, color_box, -1)
        if track_id is not None:
            cv2.putText(annotated_frame, f"{cls_name} ID {track_id}", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

        # Verificar em quais ROIs está
        for i, ROI in enumerate(ROIS):
            x_r, y_r, w_r, h_r = ROI
            dentro_roi = (x_r < cx < x_r + w_r) and (y_r < cy < y_r + h_r)

            if dentro_roi and track_id is not None:
                # Marca flags
                if cls_name == CLASSE_ALVO:
                    roi_tem_cluster[i] = True
                    # toca beep apenas 1x por ID por ROI
                    if track_id is not None and track_id not in beep_ids_por_roi[i]:
                      beep_ids_por_roi[i].add(track_id)
                      #tocar_beep()
                     # enviar_alerta(
                     #       cluster_id=12,
                     #       roi="berco_1",
                     #      duration=3.4,
                     #      frame_id=1023
                     #   )
                    track_ids_novos[i].add(track_id)

                elif cls_name == CLASSE_ALVO_2:
                    roi_tem_mao[i] = True
    # --- Verificar se na mesma ROI existem cluster + mao_com_luva
    if camera ==1:    
        for i in range(len(ROIS)):
        
            if roi_tem_cluster[i] and roi_tem_mao[i]:
                for track_id in track_ids_novos[i]:
                    if track_id not in contador_ids:
                        contador_ids.add(track_id)
                        contador_total += 1
                        print(f"✔ Contagem: CLUSTER + MAO juntos dentro da ROI {i+1} (ID {track_id})")
                objeto_dentro_roi = True  # também zera o cronômetro
    else:
        for i in range(len(ROIS)):
        
            if roi_tem_cluster[i]:
                for track_id in track_ids_novos[i]:
                    if track_id not in contador_ids:
                        contador_ids.add(track_id)
                        contador_total += 1
                        print(f"✔ Contagem: CLUSTER dentro da ROI {i+1} (ID {track_id})")
                objeto_dentro_roi = True  # também zera o cronômetro


    # --- Atualizar cronômetro ---
    tempo_atual = time.time()
    delta = tempo_atual - ultimo_tempo
    ultimo_tempo = tempo_atual

    # --- Atualizar tempos de permanência por ROI ---
    for i in range(len(ROIS)):
        if roi_tem_cluster[i]:
            
     

            tempo_cluster_roi[i] += delta
        else:
            tempo_cluster_roi[i] = 0.0
            


    if objeto_dentro_roi:
        tempo_sem_objeto = 0.0  # zera se houver algo dentro
        
    else:
        tempo_sem_objeto += delta  # incrementa enquanto não houver
        
    
    # --- Exibir mensagem se passar de 3s ---
    for i, ROI in enumerate(ROIS):
        if tempo_cluster_roi[i] >= TEMPO_REMOCAO_CLUSTER:
            
            x, y, w, h = ROI
            cv2.putText(
                annotated_frame,
                "CLUSTER AGUARDANDO REMOCAO",
                (x + 5, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    # --- Desenhar ROIs semi-transparentes ---
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

    # --- Total e cronômetro ---
    cv2.putText(annotated_frame, f"Total {CLASSE_ALVO} testados: {contador_total}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"Inatividade operacional: {tempo_sem_objeto:.1f}s", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2)
    

    if any(roi_tem_cluster):
        cluster_no_berco = True
    else:
        cluster_no_berco = False


    return annotated_frame, tempo_sem_objeto, cluster_no_berco, contador_total


def monitor_movimento_carrinho(frame, camera, programa):
    """
    Recebe um frame já obtido externamente, realiza detecção + rastreamento,
    cronometra tempo sem objetos dentro da ROI e retorna annotated_frame.
    """
    
    # tipo 1: conta apenas se cluster + mao_com_luva juntos
    # tipo 2: conta apenas se cluster sozinho
    global contador_ids, contador_total_carrinho, prev_time, tempo_sem_objeto, ultimo_tempo

    # --- Ler ROIs do INI ---
    ROIS = []
    PATH = f"config_{camera}_{programa}.ini"
    config.read(PATH)
    n_rois = config.getint('Ferramentas', 'n_rois')
    for i in range(1, n_rois + 1):
        section = f'ROI{i}'
        if section in config:
            x = int(config[section]['x'])
            y = int(config[section]['y'])
            w = int(config[section]['width'])
            h = int(config[section]['height'])
            ROIS.append((x, y, w, h))
    global tempo_cluster_roi

    if len(tempo_cluster_roi) != len(ROIS):
        tempo_cluster_roi = [0.0] * len(ROIS)
    
    

    # --- Detecção + rastreamento ---
    results = model.track(frame, persist=True, verbose=False)
    if not results:
        return frame

    res = results[0]
    annotated_frame = frame.copy()
    boxes = res.boxes
    names = model.names

    # Flag para saber se há algum objeto da classe-alvo dentro da ROI
    objeto_dentro_roi = False

   # --- Flags por ROI ---
    roi_tem_cluster = [False] * len(ROIS)
    roi_tem_mao = [False] * len(ROIS)
    track_ids_novos = [set() for _ in ROIS]

    # --- Processar todas as detecções ---
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        track_id = int(box.id[0]) if box.id is not None else None

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Desenho padrão (opcional)
        color_box = (255, 255, 0) if cls_name == CLASSE_ALVO else (255, 0, 255)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_box, 2)
        cv2.circle(annotated_frame, (cx, cy), 4, color_box, -1)
        if track_id is not None:
            cv2.putText(annotated_frame, f"{cls_name} ID {track_id}", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

        # Verificar em quais ROIs está
        for i, ROI in enumerate(ROIS):
            x_r, y_r, w_r, h_r = ROI
            dentro_roi = (x_r < cx < x_r + w_r) and (y_r < cy < y_r + h_r)

            if dentro_roi and track_id is not None:
                # Marca flags
                if cls_name == CLASSE_ALVO:
                    roi_tem_cluster[i] = True
                   
                    track_ids_novos[i].add(track_id)

                elif cls_name == CLASSE_ALVO_2:
                    roi_tem_mao[i] = True
    # --- Verificar se na mesma ROI existem cluster + mao_com_luva
    if camera ==1:    
        for i in range(len(ROIS)):
        
            if roi_tem_cluster[i] and roi_tem_mao[i]:
                for track_id in track_ids_novos[i]:
                    if track_id not in contador_ids:
                        contador_ids.add(track_id)
                        contador_total_carrinho += 1
                        print(f"✔ Contagem: CLUSTER + MAO juntos dentro da ROI {i+1} (ID {track_id})")
                objeto_dentro_roi = True  # também zera o cronômetro
    else:
        for i in range(len(ROIS)):
        
            if roi_tem_cluster[i]:
                for track_id in track_ids_novos[i]:
                    if track_id not in contador_ids:
                        contador_ids.add(track_id)
                        contador_total_carrinho += 1
                        print(f"✔ Contagem: CLUSTER dentro da ROI {i+1} (ID {track_id})")
                objeto_dentro_roi = True  # também zera o cronômetro


    # --- Atualizar cronômetro ---
    tempo_atual = time.time()
    delta = tempo_atual - ultimo_tempo
    ultimo_tempo = tempo_atual

    # --- Atualizar tempos de permanência por ROI ---
    for i in range(len(ROIS)):
        if roi_tem_cluster[i]:
            


            tempo_cluster_roi[i] += delta
        else:
            tempo_cluster_roi[i] = 0.0
            


    if objeto_dentro_roi:
        tempo_sem_objeto = 0.0  # zera se houver algo dentro
        
    else:
        tempo_sem_objeto += delta  # incrementa enquanto não houver
        
    
    # --- Exibir mensagem se passar de 3s ---
    for i, ROI in enumerate(ROIS):
        if tempo_cluster_roi[i] >= TEMPO_REMOCAO_CLUSTER:
            
            x, y, w, h = ROI
            cv2.putText(
                annotated_frame,
                "CLUSTER AGUARDANDO REMOCAO",
                (x + 5, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    # --- Desenhar ROIs semi-transparentes ---
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

    # --- Total e cronômetro ---
    cv2.putText(annotated_frame, f"Total {CLASSE_ALVO} testados: {contador_total_carrinho}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"Inatividade operacional: {tempo_sem_objeto:.1f}s", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2)

    return annotated_frame , contador_total_carrinho



####################################### usar camera 3 #############################################################
def monitor_movimento_pessoas(frame, camera, programa):
    """
    Recebe um frame já obtido externamente, realiza detecção + rastreamento,
    cronometra tempo sem objetos dentro da ROI e retorna annotated_frame.
    """

    # tipo 1: conta apenas se cluster + mao_com_luva juntos
    # tipo 2: conta apenas se cluster sozinho
    global contador_ids, contador_total_pessoas, prev_time, tempo_sem_objeto, ultimo_tempo

    # --- Ler ROIs do INI ---
    ROIS = []
    PATH = f"config_{camera}_{programa}.ini"
    config.read(PATH)
    n_rois = config.getint('Ferramentas', 'n_rois')
    for i in range(1, n_rois + 1):
        section = f'ROI{i}'
        if section in config:
            x = int(config[section]['x'])
            y = int(config[section]['y'])
            w = int(config[section]['width'])
            h = int(config[section]['height'])
            ROIS.append((x, y, w, h))
    global tempo_cluster_roi

    if len(tempo_cluster_roi) != len(ROIS):
        tempo_cluster_roi = [0.0] * len(ROIS)
    
    

    # --- Detecção + rastreamento ---
    results = model_pose.track(frame, persist=True, verbose=False)
    if not results:
        return frame

    res = results[0]
    annotated_frame = frame.copy()
    boxes = res.boxes
    names = model_pose.names

    # Flag para saber se há algum objeto da classe-alvo dentro da ROI
    objeto_dentro_roi = False

   # --- Flags por ROI ---
    roi_tem_cluster = [False] * len(ROIS)
    roi_tem_mao = [False] * len(ROIS)
    roi_tem_pessoas = [False] * len(ROIS)
    track_ids_novos = [set() for _ in ROIS]

    # --- Processar todas as detecções ---
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        track_id = int(box.id[0]) if box.id is not None else None

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Desenho padrão (opcional)
        color_box = (255, 255, 0) if cls_name == CLASSE_ALVO else (255, 0, 255)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_box, 2)
        cv2.circle(annotated_frame, (cx, cy), 4, color_box, -1)
        if track_id is not None:
            cv2.putText(annotated_frame, f"{cls_name} ID {track_id}", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

        # Verificar em quais ROIs está
        for i, ROI in enumerate(ROIS):
            x_r, y_r, w_r, h_r = ROI
            dentro_roi = (x_r < cx < x_r + w_r) and (y_r < cy < y_r + h_r)

            if dentro_roi and track_id is not None:
                # Marca flags
                if cls_name == CLASSE_ALVO_3:
                    roi_tem_pessoas[i] = True

                    track_ids_novos[i].add(track_id)

    # --- Verificar se na mesma ROI existem cluster + mao_com_luva
      
    for i in range(len(ROIS)):
        
        if roi_tem_pessoas[i]:
            for track_id in track_ids_novos[i]:
                if track_id not in contador_ids:
                    contador_ids.add(track_id)
                    contador_total_pessoas += 1
                    print(f"✔ Contagem: Pessoas dentro da ROI {i+1} (ID {track_id})")
          #  objeto_dentro_roi = True  # também zera o cronômetro
  
    # --- Desenhar ROIs semi-transparentes ---
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

    # --- Total de pessoas ---
    cv2.putText(annotated_frame, f"Total {CLASSE_ALVO_3} : {contador_total_pessoas}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return annotated_frame , contador_total_pessoas





import requests
import datetime

# URL do webhook do n8n
WEBHOOK_URL = "http://localhost:5678/webhook/cluster-alert"

def enviar_alerta(cluster_id, roi, duration, frame_id):
    payload = {
        "event": "cluster_detected",
        "cluster_id": cluster_id,
        "roi": roi,
        "duration": duration,
        "timestamp": datetime.datetime.now().isoformat(),
        "extra": {
            "frame_id": frame_id
        }
    }

    try:
        response = requests.post(WEBHOOK_URL, json=payload)
        response.raise_for_status()
        print("Alerta enviado com sucesso para o n8n!")
        print("Resposta:", response.text)

    except Exception as e:
        print("Erro ao enviar alerta:", e)


#modelo_cabo = YOLO("cabo_flex.pt")

def analise_cabo_flex(modelo_cabo, classe_ok, classe_nok, imagem, coordenadas, tolerancia=0.7):
    """
    Analisa cabo flex usando YOLO em uma ROI específica

    :param imagem: frame OpenCV (BGR)
    :param coordenadas: (x, y, w, h) da ROI
    :param tolerancia: confiança mínima para aceitar a predição
    :return: status (True=OK / False=NOK), classe, confiança
    """
    annotated =imagem
    x, y, w, h = coordenadas

    # --- Proteção de limites ---
    h_img, w_img = imagem.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)

    roi = imagem[y:y+h, x:x+w]

    if roi.size == 0:
        return False, "roi_invalida", 0.0

    # --- Inferência YOLO ---
    results = modelo_cabo(
        roi,
        conf=tolerancia,
        iou=0.5,
        verbose=False
    )

    melhor_conf = 0.0
    melhor_classe = "nenhum"
    status = False  # default seguro = NOK
    # Inferência
    results = modelo_cabo(imagem, conf=0.25, verbose=True)
    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            classe = modelo_cabo.names[cls_id]

            # Guarda melhor detecção (para debug)
            if conf > melhor_conf:
                melhor_conf = conf
                melhor_classe = classe

            # Regra crítica
            if classe == classe_nok and conf >= 0.5:
                #annotated = results[0].plot()
                return False
                
            if classe == classe_ok and conf >= 0.5:
                status = True
    #annotated = results[0].plot()
    return status

