

import cv2
import threading
import os
from flask_cors import CORS

import face_recognition
import numpy as np
import time
from datetime import datetime

from db import get_db

matricula =""

PASTA_ROSTOS = 'rostos_cadastrados_cpu'
# Dicionário de controle por câmera
cameras = {}
def carregar_rostos(pasta=PASTA_ROSTOS):
    rostos_conhecidos = []
    nomes = []
    if not os.path.exists(pasta):
        return rostos_conhecidos, nomes

    for arquivo in os.listdir(pasta):
        if arquivo.endswith(".npy"):
            rostos_conhecidos.append(np.load(os.path.join(pasta, arquivo)))
            nomes.append(os.path.splitext(arquivo)[0])
    return rostos_conhecidos, nomes

def carregar_rostos_mysql():
    """
    Carrega embeddings e dados dos colaboradores direto do MySQL.
    """
    try:
        db = get_db()
        cursor = db.cursor()

        cursor.execute("""
            SELECT matricula, nome, setor, funcao, caminho_imagem, caminho_embedding
            FROM colaboradores
        """)

        rostos_conhecidos = []
        nomes = []
        matriculas = []
        setores = []
        funcoes = []
        imagens = []

        for (matricula, nome, setor, funcao, caminho_img, caminho_npy) in cursor.fetchall():
            if os.path.exists(caminho_npy):
                embedding = np.load(caminho_npy)
                rostos_conhecidos.append(embedding)
                nomes.append(nome)
                matriculas.append(matricula)
                setores.append(setor)
                funcoes.append(funcao)
                imagens.append(caminho_img)

        return rostos_conhecidos, nomes, matriculas, setores, funcoes, imagens

    except Exception as e:
        print("Erro ao carregar rostos do MySQL:", e)
        return [], [], [], [], [], []

# Fora da função
#ROSTOS_CONHECIDOS, NOMES = carregar_rostos()

ROSTOS_CONHECIDOS, NOMES, MATRICULAS, SETORES, FUNCOES, IMAGENS = carregar_rostos_mysql()

def atualizar_rostos():
    global ROSTOS_CONHECIDOS, NOMES, MATRICULAS, SETORES, FUNCOES, IMAGENS
    ROSTOS_CONHECIDOS, NOMES, MATRICULAS, SETORES, FUNCOES, IMAGENS = carregar_rostos_mysql()



process_frame = True   # controle global (coloque fora da função)
ultimo_nome = ""

def reconhecer_api(frame, scale=0.5):
    """
    Função otimizada de reconhecimento facial.
    - Reduz o tamanho do frame antes de processar (4x mais rápido)
    - Processa apenas 1 a cada 2 frames
    - Usa dados carregados em memória
    """
    global ultimo_nome, matricula
    global process_frame
    if frame is None:
        return frame

    # Se não for o frame que deve ser processado, retorna o mesmo frame
    #if not process_frame:
     #   process_frame = True
      #  return frame

    process_frame = False

    try:
        rostos_conhecidos = ROSTOS_CONHECIDOS
        nomes = NOMES

        if not rostos_conhecidos:
            return frame

        # === 1) Reduz frame antes de processar ===
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # === 2) Detecta rostos e gera encodings ===
        
        rostos = face_recognition.face_locations(rgb_small, model='hog')
        encodings = face_recognition.face_encodings(rgb_small, rostos)

        annotated_frame = frame.copy()

        for (top, right, bottom, left), cod in zip(rostos, encodings):

            # Reescala coordenadas para o frame original
            # Reescala coordenadas para o frame original
            factor = 1.0 / scale
            top    = int(top * factor)
            right  = int(right * factor)
            bottom = int(bottom * factor)
            left   = int(left * factor)


            # === 3) Calcula distância para rostos conhecidos ===
            distancias = face_recognition.face_distance(rostos_conhecidos, cod)
            min_dist = np.min(distancias)

           

            if min_dist < 0.5:
                index = np.argmin(distancias)
                nome = NOMES[index]
                matricula = MATRICULAS[index]
                setor = SETORES[index]
                funcao = FUNCOES[index]
                foto_path = IMAGENS[index]

                confianca = float(1 - min_dist)
                cor = (0, 255, 0)
                label = f"{nome} ({confianca*100:.1f}%)"
                if nome != ultimo_nome:
                    ultimo_nome = nome
                    print(f"[{datetime.now()}] Reconhecido: {nome} - Matrícula: {matricula} - Setor: {setor} - Função: {funcao} - Confiança: {confianca*100:.2f}%")
                    tentar_saudacao(matricula)
            else:
                nome = "Desconhecido"
                foto_path = None
                cor = (0, 0, 255)
                label = nome


            # === 4) Aumenta a ROI (melhor visualização) ===
            expand = 20
            top = max(0, top - expand)
            right = min(annotated_frame.shape[1], right + expand)
            bottom = min(annotated_frame.shape[0], bottom + expand)
            left = max(0, left - expand)

            # === 5) Desenha retângulo ===
            cv2.rectangle(annotated_frame, (left, top), (right, bottom), cor, 2)

            # ==== 6) Fundo da legenda ====
            cv2.rectangle(
                annotated_frame,
                (left, bottom - 30),
                (right, bottom),
                cor,
                cv2.FILLED
            )

            cv2.putText(
                annotated_frame,
                label,
                (left + 6, bottom - 8),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (0, 0, 0),
                1
            )

            # === 7) Mostra miniatura da foto cadastrada ===
            #foto_path = os.path.join("rostos_cadastrados", f"{nome}.jpg")
            index = np.argmin(distancias)
            foto_path = IMAGENS[index]

            if nome != "Desconhecido" and os.path.exists(foto_path):
                img_ref = cv2.imread(foto_path)
                img_ref = cv2.resize(img_ref, (120, 120))

                x_offset = annotated_frame.shape[1] - 130
                y_offset = 10
                annotated_frame[
                    y_offset:y_offset+img_ref.shape[0],
                    x_offset:x_offset+img_ref.shape[1]
                ] = img_ref

        return annotated_frame, ultimo_nome,matricula

    except Exception as e:
        print(f"Erro em reconhecer_api: {e}")
        return frame


from datetime import date



# Marca o último disparo de cada alerta
ultimo_alerta = {
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
        # decide qual áudio tocar
        if os.path.exists(audio_file):
            arquivo = audio_file
        elif os.path.exists("audio/audio_generico.wav"):
            arquivo = "audio/audio_generico.wav"
        else:
            # fallback final: beep do sistema (sem arquivo)
            subprocess.Popen(
                ["aplay", "/usr/share/sounds/alsa/Front_Center.wav"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return

        subprocess.Popen(
            ["aplay", arquivo],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    threading.Thread(target=play, daemon=True).start()

def tentar_saudacao_thread(matricula):
    try:
        db = get_db()          # conexão NOVA
        cursor = db.cursor()

        hoje = date.today()

        sql = """
        INSERT IGNORE INTO saudacoes_operadores (matricula, data)
        VALUES (%s, %s)
        """

        cursor.execute(sql, (matricula, hoje))
        db.commit()

        if cursor.rowcount == 1:
            if pode_disparar("saudacao",5):
                            print("[ALERTA] Saudação de operador detectada!")
                            tocar_beep_NEW(f"audio/{matricula}.wav")

    except Exception as e:
        print(f"[SAUDACAO][ERRO] {e}")

    finally:
        try:
            cursor.close()
            db.close()
        except:
            pass

def tentar_saudacao(matricula):
    if not matricula:
        return

    threading.Thread(
        target=tentar_saudacao_thread,
        args=(matricula,),
        daemon=True
    ).start()




