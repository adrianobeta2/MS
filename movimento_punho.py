from ultralytics import YOLO
import cv2
import time

# ================= CONFIGURAÇÕES =================
MODEL_PATH = "yolov8n-pose.pt"
FRAMES_CONSECUTIVOS = 5

PULSO_ESQ = 9
PULSO_DIR = 10

# ================= MODELO GLOBAL =================
model = YOLO(MODEL_PATH)


def cria_detector_ergonomia():
    """
    Retorna uma função que avalia ergonomia por frame.
    Entrada: frame
    Saída: status ("OK" | "NAO_OK"), frame_anotado
    """

    contador_ok = 0
    prev_time = 0

    def avaliar(frame):
        nonlocal contador_ok, prev_time

        annotated = frame.copy()
        resultados = model(frame, verbose=False)

        pulso_esq = False
        pulso_dir = False
        pessoa_detectada = False

        for r in resultados:
            if r.keypoints is None:
                continue

            for kpts in r.keypoints.xy:
                pessoa_detectada = True

                pe = kpts[PULSO_ESQ]
                pd = kpts[PULSO_DIR]

                if pe[0] > 0 and pe[1] > 0:
                    pulso_esq = True
                    cv2.circle(annotated, (int(pe[0]), int(pe[1])), 6, (0, 255, 0), -1)

                if pd[0] > 0 and pd[1] > 0:
                    pulso_dir = True
                    cv2.circle(annotated, (int(pd[0]), int(pd[1])), 6, (0, 255, 0), -1)

        # ===== LÓGICA ERGONÔMICA =====
        if pessoa_detectada and pulso_esq and pulso_dir:
            contador_ok += 1
        else:
            contador_ok = 0

        ergonomia_ok = contador_ok >= FRAMES_CONSECUTIVOS
        status = "OK" if ergonomia_ok else "NAO_OK"

        # ===== VISUAL =====
        texto = "ERGONOMIA OK" if ergonomia_ok else "USE OS DOIS BRACOS"
        cor = (0, 255, 0) if ergonomia_ok else (0, 0, 255)

        cv2.putText(
            annotated,
            texto,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            cor,
            3
        )

        # FPS (opcional)
        agora = time.time()
        fps = 1 / (agora - prev_time) if prev_time else 0
        prev_time = agora

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (30, annotated.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        return status, annotated

    return avaliar
