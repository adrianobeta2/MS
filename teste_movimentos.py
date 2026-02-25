import cv2
from monitorar_movimentos import monitor_movimento, monitor_movimento_carrinho, monitor_movimento_pessoas  # ajuste se necessário

from db import get_db

VIDEO_PATH = "saida.mp4"
OUTPUT_PATH = "output_annotated.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

# Pegando propriedades do vídeo de entrada
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ================================   
# AUMENTAR RESOLUÇÃO DO VÍDEO
# ================================
scale = 1  # 2x maior, pode alterar
new_w = width * scale
new_h = height * scale

# Criando o gravador de vídeo com a nova resolução
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (new_w, new_h))

# ================================
# AUMENTAR JANELA NA TELA
# ================================
cv2.namedWindow("Teste MP4", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Teste MP4", new_w, new_h)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo.")
        break

    # Processa o frame
    annotated, tempo_sem_objeto, cluster_no_berco, contador_total_clusters_berco, tempo_cluster_na_roi, tempo_berco_1, tempo_berco_2 = monitor_movimento(frame,1,1)
    #annotated, total_clusters_carrinho = monitor_movimento_carrinho(frame,1,1)
    #annotated, contador_total_pessoas = monitor_movimento_pessoas(frame,3,1)
    # Redimensiona para salvar e exibir
    annotated_resized = cv2.resize(annotated, (new_w, new_h))

    # Salva no arquivo
    writer.write(annotated_resized)

    # Exibe
    cv2.imshow("Teste MP4", annotated_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"Vídeo salvo com sucesso em: {OUTPUT_PATH}")

