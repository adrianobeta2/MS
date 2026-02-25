import cv2
import face_recognition
import os
import numpy as np
import time

PASTA_ROSTOS = 'rostos_cadastrados'

def cadastrar_rosto(nome, pasta=PASTA_ROSTOS):
    if not os.path.exists(pasta):
        os.makedirs(pasta)

    cap = cv2.VideoCapture(0)
    print("\n[INFO] Iniciando a captura da webcam...")
    print("[INFO] Ajuste seu rosto na imagem.")
    print("[INFO] Pressione 'c' para capturar ou 'q' para cancelar.\n")

    start_time = time.time()
    timeout = 15  # segundos

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERRO] Não foi possível capturar imagem da webcam.")
            break

        cv2.imshow("Pressione 'c' para capturar o rosto", frame)

        key = cv2.waitKey(100) & 0xFF
        if key == ord('c'):
            rosto = face_recognition.face_encodings(frame)
            if rosto:
                np.save(os.path.join(pasta, nome + ".npy"), rosto[0])
                print(f"[SUCESSO] Rosto de '{nome}' salvo com sucesso!\n")
            else:
                print("[ERRO] Nenhum rosto detectado. Tente novamente.\n")
            break
        elif key == ord('q'):
            print("[INFO] Cadastro cancelado.\n")
            break

        if time.time() - start_time > timeout:
            print("[INFO] Tempo limite atingido. Encerrando.\n")
            break

    cap.release()
    cv2.destroyAllWindows()

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

def reconhecer_rosto():
    rostos_conhecidos, nomes = carregar_rostos()
    if not rostos_conhecidos:
        print("[AVISO] Nenhum rosto cadastrado ainda!\n")
        return

    cap = cv2.VideoCapture(0)
    print("\n[INFO] Reconhecimento facial iniciado. Pressione 'q' para sair.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERRO] Não foi possível acessar a webcam.")
            break

        rostos = face_recognition.face_locations(frame)
        codificacoes = face_recognition.face_encodings(frame, rostos)

        for (top, right, bottom, left), cod in zip(rostos, codificacoes):
            distancias = face_recognition.face_distance(rostos_conhecidos, cod)
            min_dist = np.min(distancias)
            if min_dist < 0.5:
                index = np.argmin(distancias)
                nome = nomes[index]
            else:
                nome = "Desconhecido"

            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, nome, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        cv2.imshow("Reconhecimento Facial", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def menu():
    while True:
        print("=== SISTEMA FACE ID ===")
        print("1. Cadastrar novo rosto")
        print("2. Reconhecer rosto")
        print("3. Sair")
        opcao = input("Escolha uma opção: ")

        if opcao == '1':
            nome = input("Digite o nome da pessoa: ").strip()
            if nome:
                cadastrar_rosto(nome)
            else:
                print("[ERRO] Nome inválido.\n")
        elif opcao == '2':
            reconhecer_rosto()
        elif opcao == '3':
            print("Encerrando o sistema.")
            break
        else:
            print("[ERRO] Opção inválida!\n")

if __name__ == "__main__":
    menu()
