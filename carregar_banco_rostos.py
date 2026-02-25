import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

PASTA_ROSTOS = "rostos_cadastrados"

def carregar_banco():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0)  # GPU

    embeddings = []
    nomes = []

    for arq in os.listdir(PASTA_ROSTOS):
        path = os.path.join(PASTA_ROSTOS, arq)

        # Caso JPG → extrai embedding
        if arq.lower().endswith(".jpg"):
            img = cv2.imread(path)
            faces = app.get(img)

            if len(faces) == 0:
                print(f"⚠ Nenhum rosto detectado em {arq}")
                continue

            emb = faces[0].embedding

            # garante tamanho coerente
            if emb.shape[0] < 200:
                print(f"⚠ Ignorando {arq}: embedding muito pequeno ({emb.shape[0]})")
                continue

            embeddings.append(emb)
            nomes.append(os.path.splitext(arq)[0])

        # Caso NPY → verificar se é do InsightFace
        elif arq.lower().endswith(".npy"):
            emb = np.load(path)

            if emb.shape[0] < 200:  # elimina dlib
                print(f"⚠ Ignorando {arq}: embedding dlib incompatível ({emb.shape[0]})")
                continue

            embeddings.append(emb)
            nomes.append(os.path.splitext(arq)[0])

    embeddings = np.array(embeddings)
    return embeddings, nomes

