import cv2
import numpy as np
from insightface.app import FaceAnalysis
from carregar_banco_rostos import carregar_banco

print("🔄 Carregando banco de rostos...")
EMBEDS, NOMES = carregar_banco()
print(f"✔ Banco carregado: {len(NOMES)} pessoas")

# Inicializa modelo na GPU
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)   # GPU RTX

def reconhecer_api(frame):

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        emb = face.embedding

        # Distâncias
        distancias = np.linalg.norm(EMBEDS - emb, axis=1)
        idx = np.argmin(distancias)
        dist = distancias[idx]

        if dist < 0.8:   # limiar ideal p/ InsightFace
            nome = NOMES[idx]
            cor = (0,255,0)
            label = f"{nome} ({(1-dist)*100:.1f}%)"

            # miniatura
            foto = f"rostos_cadastrados/{nome}.jpg"
            if os.path.exists(foto):
                img = cv2.imread(foto)
                img = cv2.resize(img, (120,120))
                frame[10:130, frame.shape[1]-130:frame.shape[1]-10] = img

        else:
            nome = "Desconhecido"
            cor = (0,0,255)
            label = nome

        # Desenha ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)

        # Fundo texto
        cv2.rectangle(frame, (x1, y2 - 30), (x2, y2), cor, cv2.FILLED)

        # Texto
        cv2.putText(frame, label, (x1 + 5, y2 - 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 1)

    return frame
