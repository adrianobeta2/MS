import cv2
import time
import csv
import os
import configparser
from ultralytics import YOLO
import threading

from collections import deque

from flask_cors import CORS


import numpy as np
import time
from datetime import datetime, date 










#rom movimento_punho_nrois import ROIS

JANELA = 10
historico_luvas = deque(maxlen=JANELA)

# === CONFIGURAÇÕES GERAIS ===
MODEL_PATH = "best_8_n.pt"
#MODEL_PATH_POSE = "yolov8n-pose.pt"
MODEL_PATH_POSE = "operadores.pt"

CLASSE_ALVO = "cluster"
CLASSE_ALVO_2 = "maos_com_luva"
CLASSE_ALVO_3 = "operador"
CLASSE_ALVO_4 = "berco_1"
CLASSE_ALVO_5 = "berco_2"
CLASSE_TECNICO = "tecnico"

tempo_classe_berco_1 = 0.0
tempo_classe_berco_2 = 0.0

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
contador_ids_pessoas = set()
contador_ids_carrinho = set()

contador_total = 0
contador_total_carrinho = 0
contador_total_pessoas = 0
cluster_no_berco = False

prev_time = 0
prev_time_carrinho = 0
prev_time_pessoas = 0

tempo_sem_objeto = 0.0
tempo_sem_objeto_carrinho = 0.0
tempo_sem_objeto_pessoas = 0.0

tempo_objeto_dentro_roi = 0.0
tempo_objeto_dentro_roi_carrinho = 0.0
tempo_objeto_dentro_roi_pessoas = 0.0

ultimo_tempo = time.time()
ultimo_tempo_carrinho = time.time()
ultimo_tempo_pessoas = time.time()


ultimo_frame_berco_1 = time.time()
ultimo_frame_berco_2 = time.time()

ultimo_frame_berco_1 = 0.0
ultimo_frame_berco_2 = 0.0

TIMEOUT_BERCO = 0.5  # segundos tolerados sem detecção



# === MODELO YOLO ===
#model = YOLO(MODEL_PATH)
#model_pose = YOLO(MODEL_PATH_POSE)

import torch

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO(MODEL_PATH).to(device)
model_pose = YOLO(MODEL_PATH_POSE).to(device)

print("YOLO device model:", model.device)
print("YOLO device model_pose:", model_pose.device)


# controle do tempo de permanência de cluster por ROI
tempo_cluster_roi = []
tempo_cluster_roi_carrinho = []
tempo_cluster_roi_pessoas = []

#def tocar_beep():
 #  threading.Thread(target=playsound, args=("beep_trim.mp3",), daemon=True).start()
import subprocess
import threading

def tocar_beep():
    def play():
        subprocess.Popen(
            ["aplay", "beep_trim.wav"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    threading.Thread(target=play, daemon=True).start()


# Marca o último disparo de cada alerta
ultimo_alerta = {
    "tecnico_reparo": 0,
    "cluster_berco": 0,
    "ergonomia": 0,
    "saudacao": 0,
}
COOLDOWN_ALERTA = 5 # segundos
def pode_disparar(chave,cooldown):
    """Verifica se já passou o tempo mínimo para novo alerta"""
    agora = time.time()
    if agora - ultimo_alerta[chave] >= cooldown:
        ultimo_alerta[chave] = agora
        return True
    return False

import subprocess
def tocar_beep_NEW(audio_file="beep_trim.wav"):
    def play():
        subprocess.Popen(
            ["aplay", audio_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    threading.Thread(target=play, daemon=True).start()



def tocar_audio(som):
    def play():
        subprocess.Popen(
            ["aplay", som],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    threading.Thread(target=play, daemon=True).start()

# IDs que já tocaram beep por ROI
beep_ids_por_roi = []
tempo_tecnico_por_roi = []



def monitor_movimento(frame, camera, programa):
    """
    Recebe um frame já obtido externamente, realiza detecção + rastreamento,
    cronometra tempo sem objetos dentro da ROI e retorna annotated_frame.
    """
    
    global beep_ids_por_roi

    

    # tipo 1: conta apenas se cluster + mao_com_luva juntos
    # tipo 2: conta apenas se cluster sozinho
    global contador_ids, contador_total, prev_time, tempo_sem_objeto, ultimo_tempo, tempo_objeto_dentro_roi
    global tempo_classe_berco_1, tempo_classe_berco_2, ultimo_frame_berco_1, ultimo_frame_berco_2


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
    #results = model.track(frame, persist=True, verbose=False)
    results = model.track(frame, persist=True, verbose=False, device=0)
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

    berco_1_detectado = False  #essa classe está atrelada aao adesivo dos bercos na tampa da maquina
    berco_2_detectado = False
  

    # ===== INÍCIO DO FRAME =====
    maos_ids = set()              # mãos globais (sem ROI)
    




    # --- Processar todas as detecções ---
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]

    

        if box.id is None:
            continue
        track_id = int(box.id[0])
        
        if cls_name == CLASSE_ALVO_4:  # "berco_1"
            berco_1_detectado = True

        elif cls_name == CLASSE_ALVO_5:  # "berco_2"
            berco_2_detectado = True
        
       

         # -------- mãos (SEM ROI) --------
        if cls_name == CLASSE_ALVO_2:  # mão com luva
            maos_ids.add(track_id) 

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
                      if pode_disparar("cluster_berco",5):
                            print("[ALERTA] Clusters aguardando no berco!")
                            tocar_beep_NEW("beep_trim.wav")
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

    ################## ergonomia ###################################
    
    qtd_maos = len(maos_ids)
    if qtd_maos >= 2:
    #if qtd_maos == 0:
            print(f"[OK] ROI {i}: cluster com {qtd_maos} mãos visíveis")
            #if pode_disparar("ergonomia",120):
            #                print("[ALERTA] Clusters aguardando no berco!")
            #                tocar_beep_NEW("ergonomia.wav")

       
    ###############################################################

    # --- Verificar se na mesma ROI existem cluster + mao_com_luva
    if camera ==1:    
        for i in range(len(ROIS)):
            
            
            #####################################################
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

    # --- Contador de tempo por classe ---
    if berco_1_detectado:
        tempo_classe_berco_1 += delta
        ultimo_frame_berco_1 = tempo_atual
    else:
        if tempo_atual - ultimo_frame_berco_1 > TIMEOUT_BERCO:
             tempo_classe_berco_1 = 0.0

    if berco_2_detectado:
        tempo_classe_berco_2 += delta
        ultimo_frame_berco_2 = tempo_atual
    else:
        if tempo_atual - ultimo_frame_berco_2 > TIMEOUT_BERCO:
             tempo_classe_berco_2 = 0.0


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
            tempo_objeto_dentro_roi = tempo_cluster_roi[i]
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
    
    # --- Tempo de cluster dentro da roi ---
    cv2.putText(annotated_frame, f"Tempo cluster na ROI: {tempo_objeto_dentro_roi:.1f}s", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2)
    
    cv2.putText(
    annotated_frame,
    f"Tempo berco_1: {tempo_classe_berco_1:.1f}s",
    (100, 150),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 0, 0),
    2)

    cv2.putText(
        annotated_frame,
        f"Tempo berco_2: {tempo_classe_berco_2:.1f}s",
        (460, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2)



    if any(roi_tem_cluster):
        cluster_no_berco = True
    else:
        cluster_no_berco = False


    return annotated_frame, tempo_sem_objeto, cluster_no_berco, contador_total, tempo_objeto_dentro_roi,tempo_classe_berco_1, tempo_classe_berco_2


def monitor_movimento_carrinho(frame, camera, programa):
    """
    Recebe um frame já obtido externamente, realiza detecção + rastreamento,
    cronometra tempo sem objetos dentro da ROI e retorna annotated_frame.
    """
    
    # tipo 1: conta apenas se cluster + mao_com_luva juntos
    # tipo 2: conta apenas se cluster sozinho
    global contador_ids_carrinho, contador_total_carrinho, prev_time_carrinho, tempo_sem_objeto_carrinho, ultimo_tempo_carrinho

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
    global tempo_cluster_roi_carrinho

    if len(tempo_cluster_roi_carrinho) != len(ROIS):
        tempo_cluster_roi_carrinho = [0.0] * len(ROIS)
    
    

    # --- Detecção + rastreamento ---
    #results = model.track(frame, persist=True, verbose=False)
    results = model.track(frame, persist=True, verbose=False, device=0)
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
                    if track_id not in contador_ids_carrinho:
                        contador_ids_carrinho.add(track_id)
                        contador_total_carrinho += 1
                        print(f"✔ Contagem: CLUSTER + MAO juntos dentro da ROI {i+1} (ID {track_id})")
                objeto_dentro_roi = True  # também zera o cronômetro
    else:
        for i in range(len(ROIS)):
        
            if roi_tem_cluster[i]:
                for track_id in track_ids_novos[i]:
                    if track_id not in contador_ids_carrinho:
                        contador_ids_carrinho.add(track_id)
                        contador_total_carrinho += 1
                        #print(f"✔ Contagem: CLUSTER dentro da ROI {i+1} (ID {track_id})")
                objeto_dentro_roi = True  # também zera o cronômetro


    # --- Atualizar cronômetro ---
    tempo_atual = time.time()
    delta = tempo_atual - ultimo_tempo_carrinho
    ultimo_tempo_carrinho = tempo_atual

    # --- Atualizar tempos de permanência por ROI ---
    for i in range(len(ROIS)):
        if roi_tem_cluster[i]:
            


            tempo_cluster_roi_carrinho[i] += delta
        else:
            tempo_cluster_roi_carrinho[i] = 0.0
            


    if objeto_dentro_roi:
        tempo_sem_objeto_carrinho = 0.0  # zera se houver algo dentro
        
    else:
        tempo_sem_objeto_carrinho += delta  # incrementa enquanto não houver
        
    # --- Exibir mensagem se passar de 3s ---
    for i, ROI in enumerate(ROIS):
        if tempo_cluster_roi_carrinho[i] >= TEMPO_REMOCAO_CLUSTER:
            
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
    fps = 1 / (current_time - prev_time_carrinho) if prev_time_carrinho else 0
    prev_time_carrinho = current_time
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # --- Total e cronômetro ---
    cv2.putText(annotated_frame, f"Total de clusters no carrinho: {contador_total_carrinho}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    #cv2.putText(annotated_frame, f"Inatividade operacional: {tempo_sem_objeto_carrinho:.1f}s", (30, 90),
      #          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2)

    return annotated_frame , contador_total_carrinho



####################################### usar camera 3 #############################################################

AUDIO_TECNICO_POR_ROI = {
    0: "tecnico_op1.wav",
    1: "tecnico_op2.wav",
    2: "tecnico_op3.wav",
    3: "tecnico_op4.wav",
}
audio_tecnico_tocado = []


def monitor_movimento_pessoas(frame, camera, programa):
    """
    Recebe um frame já obtido externamente, realiza detecção + rastreamento,
    cronometra tempo sem objetos dentro da ROI e retorna annotated_frame.
    """

    # tipo 1: conta apenas se cluster + mao_com_luva juntos
    # tipo 2: conta apenas se cluster sozinho
    global contador_ids_pessoas, contador_total_pessoas, prev_time_pessoas, tempo_sem_objeto_pessoas, ultimo_tempo_pessoas, tempo_tecnico_por_roi,tempo_cluster_roi_pessoas, audio_tecnico_tocado

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

    if len(tempo_tecnico_por_roi) != len(ROIS):
       tempo_tecnico_por_roi = [0.0] * len(ROIS)


    if len(tempo_cluster_roi_pessoas) != len(ROIS):
        tempo_cluster_roi_pessoas = [0.0] * len(ROIS)

    if len(audio_tecnico_tocado) != len(ROIS):
         audio_tecnico_tocado = [False] * len(ROIS)

    
    

    # --- Detecção + rastreamento ---
    #results = model_pose.track(frame, persist=True, verbose=False)
    results = model_pose.track(frame, persist=True, verbose=False, device=0)
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
    roi_tem_tecnico = [False] * len(ROIS)

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
                elif cls_name == CLASSE_TECNICO:
                    roi_tem_tecnico[i] = True

    # --- Verificar se na mesma ROI existem cluster + mao_com_luva
      
    for i in range(len(ROIS)):
        
        if roi_tem_pessoas[i]:
            for track_id in track_ids_novos[i]:
                if track_id not in contador_ids_pessoas:
                    contador_ids_pessoas.add(track_id)
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
        
    # --- Atualizar cronômetro ---
    tempo_atual = time.time()
    delta = tempo_atual - ultimo_tempo_pessoas
    ultimo_tempo_pessoas = tempo_atual

    for i in range(len(ROIS)):
        if roi_tem_tecnico[i]:
            tempo_tecnico_por_roi[i] += delta



    # --- FPS ---
    current_time = time.time()
    fps = 1 / (current_time - prev_time_pessoas) if prev_time_pessoas else 0
    prev_time_pessoas = current_time
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # --- Total de pessoas ---
    cv2.putText(annotated_frame, f"Total {CLASSE_ALVO_3} : {contador_total_pessoas}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    for i in range(len(ROIS)):
        if roi_tem_tecnico[i]:
            x, y, w, h = ROIS[i]

            mensagem = f"Há um técnico na op  {i+1}"

            cv2.putText(
                annotated_frame,
                mensagem,
                (x + 5, y + h + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
    for i, ROI in enumerate(ROIS):
      x, y, w, h = ROI
      if roi_tem_tecnico[i]:
        if tempo_tecnico_por_roi[i] > 0:
                cv2.putText(
                    annotated_frame,
                    f"Técnico: {tempo_tecnico_por_roi[i]:.1f}s",
                    (x + 5, y + h + 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )
    for i in range(len(ROIS)):
        if roi_tem_tecnico[i]:
            if not audio_tecnico_tocado[i]:
                audio = AUDIO_TECNICO_POR_ROI.get(i)
                if audio:
                    #tocar_audio(audio)
                    audio_tecnico_tocado[i] = True
        else:
            # Reseta quando o técnico sair da ROI
            audio_tecnico_tocado[i] = False


    
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


def analise_cabo_flex(modelo_cabo, classe_ok, classe_nok, imagem, coordenadas, tolerancia=0.7, salvar_confianca=False):

    x, y, w, h = coordenadas
   
    # --- Proteção de limites ---
    h_img, w_img = imagem.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    print(modelo_cabo.device)
    # Inferência YOLO na imagem inteir
    
    results = modelo_cabo(imagem, conf=0.25, verbose=False)

    melhor_conf = 0.0
    melhor_classe = "nenhum"
    status = False

    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            classe = modelo_cabo.names[cls_id]

            # coordenadas da box detectada
            x1, y1, x2, y2 = box.xyxy[0]

            # centro da detecção
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # verificar se a detecção está dentro da ROI
            if x <= cx <= x + w and y <= cy <= y + h:

                if conf > melhor_conf:
                    melhor_conf = conf
                    melhor_classe = classe
                
                print(f"Detecção dentro da ROI: classe={classe}, conf={conf:.2f}")

                if salvar_confianca:
                  salvar_confianca_csv(classe, conf, classe_ok, classe_nok)

                # regra NOK tem prioridade
                if classe == classe_nok and conf >= 0.5:
                    return False

                if classe == classe_ok and conf >= 0.7:
                    status = True

    return status


import csv
import os
from datetime import datetime

CSV_FILE = "nivel_conf.csv"
LIMITE_LINHAS = 500

def salvar_confianca_csv(classe, conf, classe_ok, classe_nok):
    
    # Verifica quantas linhas já existem e limpa se atingiu o limite
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r") as f:
            total_linhas = sum(1 for _ in f) - 1  # desconta o cabeçalho
        
        if total_linhas >= LIMITE_LINHAS:
            os.remove(CSV_FILE)  # apaga o arquivo para recriar do zero

    # Cria o arquivo com cabeçalho se não existir
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "classe_ok_conf", "classe_nok_conf", "resultado"])

    # Captura os valores de confiança por classe
    conf_ok  = conf if classe == classe_ok  else None
    conf_nok = conf if classe == classe_nok else None

    # Determina o resultado
    if classe == classe_nok and conf >= 0.5:
        resultado = "NOK"
    elif classe == classe_ok and conf >= 0.45:
        resultado = "OK"
    else:
        resultado = "INCERTO"

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), conf_ok, conf_nok, resultado])





