import configparser
import os
from PIL import Image

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator



import random
from pathlib import Path

# Pasta das imagens de referência
IMG_DIR = os.path.join(os.path.dirname(__file__), 'static', 'imagens')

IMG_SIZE = 224




# ===== CONFIG =====
IMAGES_OK = "imagens_ok"
IMAGES_NOK = "imagens_nok"
DATASET_DIR = "dataset"



# =========================================================
# Carregar imagem + ROI (AGORA EM RGB 224x224)
# =========================================================
def carregar_imagem(caminho, coordenadas):
    x, y, w, h = coordenadas
    img = cv2.imread(caminho)

    if img is None:
        return None

    roi = img[y:y+h, x:x+w]

    if roi.size == 0:
        return None

    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi = roi.astype("float32") / 255.0
    return roi


# =========================================================
# Treinamento por câmera / programa / ROI
# =========================================================
def treinar(camera, programa):

    config = configparser.ConfigParser()
    config.read(f'config_{camera}_{programa}.ini')
    n_rois = config.getint('Ferramentas', 'n_rois')

    # Templates OK / NOK (mantidos)
    template_images_ok = [f'cam{camera}_ref_programa{programa}_OK_{i if i>0 else ""}.png' for i in range(0, 100)]
    template_images_nok = [f'cam{camera}_ref_programa{programa}_NOK_{i if i>0 else ""}.png' for i in range(0, 100)]

    for i in range(1, n_rois + 1):

        section_name = f'ROI{i}'
        if section_name not in config:
            continue

        x_coord = int(config[section_name].get('x', config[section_name].get('x_anterior')))
        y_coord = int(config[section_name].get('y', config[section_name].get('y_anterior')))
        w_coord = int(config[section_name].get('width', config[section_name].get('width_anterior')))
        h_coord = int(config[section_name].get('height', config[section_name].get('height_anterior')))

        coordenadas = (x_coord, y_coord, w_coord, h_coord)

        X, Y = [], []

        # OK = 1
        for fname in template_images_ok:
            caminho = os.path.join(IMG_DIR, fname)
            if os.path.exists(caminho):
                img = carregar_imagem(caminho, coordenadas)
                if img is not None:
                    X.append(img)
                    Y.append(1)

        # NOK = 0
        for fname in template_images_nok:
            caminho = os.path.join(IMG_DIR, fname)
            if os.path.exists(caminho):
                img = carregar_imagem(caminho, coordenadas)
                if img is not None:
                    X.append(img)
                    Y.append(0)

        X = np.array(X)
        Y = np.array(Y)

        if len(X) < 10:
            print(f"[WARN] Poucas imagens na {section_name}, pulando...")
            continue

        # =========================================================
        # Data Augmentation (industrial)
        # =========================================================
        datagen = ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.05,
            brightness_range=[0.8, 1.2]
        )

        # =========================================================
        # MODELO — TRANSFER LEARNING
        # =========================================================
        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        )

        base_model.trainable = False  # congela backbone

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        output = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=base_model.input, outputs=output)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # =========================================================
        # TREINAMENTO
        # =========================================================
        model.fit(
            datagen.flow(X, Y, batch_size=4),
            epochs=30,
            verbose=1
        )

        # =========================================================
        # SALVAR MODELO POR ROI
        # =========================================================
        modelo_path = f"modelo_cam{camera}_prog{programa}_{section_name}.keras"
        model.save(modelo_path)

        print(f"✅ Modelo salvo: {modelo_path}")

    return "Treinamento finalizado"


### ################ criar dataset no formato YOLO (imagens + labels) para cada câmera/programa/ROI alem do modelo ################

#DATASET_DIR = f"dataset/yamls"  # pasta base do dataset

TRAIN_SPLIT = 0.8



# DENFINICAO DE CLASSES
CLASSES = {
    "ok": 0,
    "nok": 1
}


def get_dataset_dir(camera, programa):
    return Path("dataset", f"cam{camera}_prog{programa}")


def arquivo_dataset(camera, programa):
    dataset_dir = get_dataset_dir(camera, programa)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    yaml = f"""path: {dataset_dir.resolve()}
train: images/train
val: images/val

names:
  0: ok
  1: nok
"""

    with open(dataset_dir / "data.yaml", "w") as f:
        f.write(yaml)

    print("✅ data.yaml criado")


def yolo_label(x, y, w, h, iw, ih, class_id):
    xc = (x + w / 2) / iw
    yc = (y + h / 2) / ih
    return f"{class_id} {xc:.6f} {yc:.6f} {w/iw:.6f} {h/ih:.6f}"




def gerar_dataset_yolo(camera, programa, incremental=False):

    config = configparser.ConfigParser()
    config.read(f'config_{camera}_{programa}.ini')

    n_rois = config.getint('Ferramentas', 'n_rois')

    template_ok = [
        f'cam{camera}_ref_programa{programa}_OK_{i if i > 0 else ""}.png'
        for i in range(0, 200)
    ]
    template_nok = [
        f'cam{camera}_ref_programa{programa}_NOK_{i if i > 0 else ""}.png'
        for i in range(0, 200)
    ]

    dataset_dir = get_dataset_dir(camera, programa)

    for p in ["images/train", "images/val", "labels/train", "labels/val"]:
        (dataset_dir / p).mkdir(parents=True, exist_ok=True)

    # ✅ Descobre quais imagens JÁ existem no dataset
    if incremental:
        imagens_existentes = set(
            f.stem.rsplit("_", 1)[0]  # remove o sufixo _idx
            for f in (dataset_dir / "images" / "train").glob("*.jpg")
        ) | set(
            f.stem.rsplit("_", 1)[0]
            for f in (dataset_dir / "images" / "val").glob("*.jpg")
        )
        print(f"[INFO] Modo incremental — {len(imagens_existentes)} imagens já no dataset")
    else:
        imagens_existentes = set()

    amostras = []

    for i in range(1, n_rois + 1):
        sec = f'ROI{i}'
        if sec not in config:
            continue

        x = int(config[sec].get('x', config[sec].get('x_anterior')))
        y = int(config[sec].get('y', config[sec].get('y_anterior')))
        w = int(config[sec].get('width', config[sec].get('width_anterior')))
        h = int(config[sec].get('height', config[sec].get('height_anterior')))
        roi = (x, y, w, h)

        for fname in template_ok:
            caminho = os.path.join(IMG_DIR, fname)
            if os.path.exists(caminho):
                nome_base = Path(fname).stem
                # ✅ Ignora se já foi processada
                if incremental and nome_base in imagens_existentes:
                    print(f"[SKIP] Já existe: {fname}")
                    continue
                amostras.append((caminho, roi, CLASSES["ok"]))

        for fname in template_nok:
            caminho = os.path.join(IMG_DIR, fname)
            if os.path.exists(caminho):
                nome_base = Path(fname).stem
                if incremental and nome_base in imagens_existentes:
                    print(f"[SKIP] Já existe: {fname}")
                    continue
                amostras.append((caminho, roi, CLASSES["nok"]))

    if not amostras:
        print("[INFO] Nenhuma imagem nova encontrada. Dataset já está atualizado.")
        return

    print(f"[INFO] {len(amostras)} novas amostras para adicionar")

    # ✅ Conta índice a partir do que já existe para não sobrescrever
    idx_offset = len(list((dataset_dir / "images" / "train").glob("*.jpg"))) + \
                 len(list((dataset_dir / "images" / "val").glob("*.jpg")))

    random.shuffle(amostras)
    split = int(len(amostras) * TRAIN_SPLIT)

    for idx, (img_path, roi, class_id) in enumerate(amostras):
        subset = "train" if idx < split else "val"

        img = cv2.imread(img_path)
        if img is None:
            continue

        h_img, w_img = img.shape[:2]
        img_name = f"{Path(img_path).stem}_{idx + idx_offset}.jpg"  # ✅ índice único

        out_img = dataset_dir / "images" / subset / img_name
        cv2.imwrite(str(out_img), img)

        x, y, w, h = roi
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = min(w, w_img - x)
        h = min(h, h_img - y)

        if w <= 0 or h <= 0:
            print(f"[WARN] ROI inválida ignorada | Cam={camera} Prog={programa}")
            continue

        label = yolo_label(x, y, w, h, w_img, h_img, class_id)
        out_lbl = dataset_dir / "labels" / subset / f"{Path(img_name).stem}.txt"
        with open(out_lbl, "w") as f:
            f.write(label + "\n")

        print(f"[CAM{camera} PROG{programa}] {subset} -> {img_name}")

    print(f"\n✅ Dataset atualizado — {len(amostras)} novas imagens adicionadas!")




import shutil
from pathlib import Path

def limpar_treino(camera, programa):
    base_treinos = Path("runs/detect/treinos")
    treino_dir = base_treinos / f"cam{camera}_prog{programa}"

    if treino_dir.exists() and treino_dir.is_dir():
        shutil.rmtree(treino_dir)
        print(f"🧹 Treino removido: {treino_dir}")
    else:
        print(f"ℹ️ Treino não existe: {treino_dir}")

import shutil

def limpar_dataset(camera, programa):
    dataset_dir = get_dataset_dir(camera, programa)

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
        print(f"🧹 Pasta removida completamente: {dataset_dir}")
    else:
        print(f"ℹ️ Pasta não existe, nada a remover: {dataset_dir}")




from ultralytics import YOLO
MODEL_BASE = "yolo8n.pt"   # leve e rápido
#DATA_YAML = "dataset/data.yaml"
EPOCHS = 50
EPOCHS_FINETUNE = 20
IMG_SIZE_YOLO = 640

from pathlib import Path
from ultralytics import YOLO
import torch


from training_state import write_status

def on_epoch_end(trainer):
    epoch = trainer.epoch + 1
    total = trainer.args.epochs  # YOLO v8+

    write_status({
        "running": True,
        "epoch": epoch,
        "total_epochs": total,
        "progress": int((epoch / total) * 100),
        "mensagem": f"Epoch {epoch}/{total}"
    })

    print(f"[CALLBACK] Epoch {epoch}/{total}")

def criar_modelo(camera, programa, img_size,modelo_base, incremental=False):
    # Diretório do dataset (Path é 100% compatível com Windows)
    dataset_dir = Path(get_dataset_dir(camera, programa))
    data_yaml = dataset_dir / "data.yaml"

    # Seleção automática de device
    device = 0 if torch.cuda.is_available() else "cpu"

    modelo_existente = Path("runs/detect/treinos") / f"cam{camera}_prog{programa}" / "weights" / "best.pt"

    if incremental and modelo_existente.exists():
        print(f"🔄 Treinamento incremental com: {modelo_existente}")
        model = YOLO(str(modelo_existente))
        epocas = EPOCHS_FINETUNE          # ✅ AQUI está a economia
       
        paciencia = 15                      
    else:
        print(f"🚀 Treinamento do zero com: {modelo_base}")
        model = YOLO(modelo_base)
        epocas = EPOCHS
        
        paciencia = 10
    from training_state import write_status

    write_status({
        "running": True,
        "epoch": 0,
        "total_epochs": epocas,
        "progress": 0,
        "mensagem": "Iniciando treinamento..."
    })
   
    model.add_callback("on_train_epoch_end", on_epoch_end)
    # Treinamento
    model.train(
        data=str(data_yaml),      
        epochs=epocas,
        imgsz=img_size,
        project="treinos",        
        name=f"cam{camera}_prog{programa}",
        batch=16,
        device=device,
        workers=0                    
    )
    write_status({
        "running": False,
        "epoch": epocas,
        "total_epochs": epocas,
        "progress": 100,
        "mensagem": "Treinamento concluído"
    })
    print("\n🎯 Treinamento finalizado!")

from pathlib import Path
from ultralytics import YOLO
import subprocess
import torch
import shutil
import sys

def exportar_modelo_trt(camera: int, programa: int, workspace: int = 4096):
    """
    Exporta best.pt -> best.onnx -> best_fp16.engine
    USANDO CLI (evita travamento no Windows)
    """

    weights_dir = Path("runs/detect/treinos") / f"cam{camera}_prog{programa}" / "weights"
    best_pt = weights_dir / "best.pt"

    if not best_pt.exists():
        raise FileNotFoundError(f"❌ best.pt não encontrado: {best_pt}")

    print(f"🔍 Modelo: {best_pt}")

   
    # ==================================
    # LOCALIZA yolo.exe DO VENV
    # ==================================
    yolo_exe = Path(sys.executable).parent / "yolo.exe"

    if not yolo_exe.exists():
        raise FileNotFoundError(f"❌ yolo.exe não encontrado em {yolo_exe}")

    # ==================================
    # EXPORTAÇÃO ONNX (CLI ISOLADO)
    # ==================================
    print("⚙️ Exportando para ONNX (CLI isolado)...")

    subprocess.run(
        [
            str(yolo_exe),
            "export",
            f"model={best_pt}",
            "format=onnx",
            "opset=12",
            "simplify=False"
        ],
        check=True
    )

    best_onnx = best_pt.with_suffix(".onnx")

    if not best_onnx.exists():
        raise RuntimeError("❌ ONNX não foi gerado")

    print(f"✅ ONNX gerado: {best_onnx}")

    # ==================================
    # TENSORRT FP16
    # ==================================
    if torch.cuda.is_available() and shutil.which("trtexec"):
        best_engine = best_pt.with_name("best_fp16.engine")

        print("🚀 Convertendo ONNX → TensorRT FP16...")

        subprocess.run(
            [
                "trtexec",
                f"--onnx={best_onnx}",
                f"--saveEngine={best_engine}",
                "--fp16",
                f"--workspace={workspace}"
            ],
            check=True
        )

        if not best_engine.exists():
            raise RuntimeError("❌ Engine TensorRT não foi gerada")

        print(f"🔥 TensorRT engine gerada: {best_engine}")
    else:
        print("ℹ️ CUDA/TensorRT não disponível — exportação limitada ao ONNX")

    print("✅ Exportação concluída com sucesso!")


def gerar_modelo(camera, programa,img_size, modelo_base, incremental=False):
    if incremental:
        print(f"🔄 Iniciando treinamento incremental para Cam={camera} Prog={programa}")
    else:
        print(f"🚀 Iniciando treinamento do zero para Cam={camera} Prog={programa}")
        limpar_dataset(camera, programa)
        limpar_treino(camera, programa)
        arquivo_dataset(camera, programa)
    gerar_dataset_yolo(camera, programa, incremental=incremental)
    criar_modelo(camera, programa, img_size,modelo_base, incremental=incremental)

### rede resnet18 ou mobilenetv2 (mais leve) para comparar com o yolo (que é mais complexo mas pode ser mais preciso e rápido na inferência)


def gerar_dataset_resnet(camera, programa):

    config = configparser.ConfigParser()
    config.read(f'config_{camera}_{programa}.ini')

    n_rois = config.getint('Ferramentas', 'n_rois')

    dataset_dir = get_dataset_dir(camera, programa)

    for subset in ["train", "val"]:
        for classe in ["ok", "nok"]:
            (dataset_dir / subset / classe).mkdir(parents=True, exist_ok=True)

    amostras = []

    for i in range(1, n_rois + 1):
        sec = f'ROI{i}'
        if sec not in config:
            continue

        x = int(config[sec].get('x'))
        y = int(config[sec].get('y'))
        w = int(config[sec].get('width'))
        h = int(config[sec].get('height'))

        roi = (x, y, w, h)

        for classe in ["OK", "NOK"]:
            for idx in range(0, 200):
                nome = f'cam{camera}_ref_programa{programa}_{classe}_{idx if idx > 0 else ""}.png'
                caminho = os.path.join(IMG_DIR, nome)

                if os.path.exists(caminho):
                    amostras.append((caminho, roi, classe.lower()))

    random.shuffle(amostras)
    split = int(len(amostras) * TRAIN_SPLIT)
    

    for idx, (img_path, roi, classe) in enumerate(amostras):

        subset = "train" if idx < split else "val"

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, y, w, h = roi
        crop = img[y:y+h, x:x+w]

        if crop.size == 0:
            continue

        img_name = f"{Path(img_path).stem}_{idx}.jpg"
        out_path = dataset_dir / subset / classe / img_name

        Image.fromarray(crop).save(out_path)

    print("✅ Dataset ResNet gerado!")
   



import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
EPOCHS_RESNET = 40

def treinar_resnet(camera, programa, img_size):

    dataset_dir = get_dataset_dir(camera, programa)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
       )
    ])


    train_dataset = datasets.ImageFolder(dataset_dir / "train", transform=transform)
    val_dataset = datasets.ImageFolder(dataset_dir / "val", transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,       # RTX 4050 aguenta
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(val_dataset, batch_size=16)

    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda")
    write_status({
        "running": True,
        "epoch": 0,
        "total_epochs": EPOCHS_RESNET,
        "progress": 0,
        "mensagem": "Iniciando treinamento ResNet..."
    })
    for epoch in range(20):
        model.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")
        progress = int(((epoch + 1) / EPOCHS_RESNET) * 100)

        write_status({
            "running": True,
            "epoch": epoch + 1,
            "total_epochs": EPOCHS_RESNET,
            "progress": progress,
            "mensagem": f"Epoch {epoch + 1}/{EPOCHS_RESNET} | Loss: {total_loss:.4f}"
        })
            
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
   
    from sklearn.metrics import classification_report
    # Avaliação final  
    labels_real, preds = avaliar_modelo(model, val_loader, device)
    from sklearn.metrics import confusion_matrix
    print("\n📊 Relatório detalhado:")
    print(classification_report(labels_real, preds, target_names=["nok", "ok"]))

    cm = confusion_matrix(labels_real, preds)
    print(cm)
    torch.save(model.state_dict(), dataset_dir / "modelo_resnet34.pth")
    print(train_dataset.class_to_idx)
    write_status({
        "running": False,
        "epoch": EPOCHS_RESNET,
        "total_epochs": EPOCHS_RESNET,
        "progress": 100,
        "mensagem": "Treinamento ResNet concluído"
    })
    print("🎯 Modelo ResNet treinado!")
    

def gerar_modelo_resnet(camera, programa,img_size):
    limpar_dataset(camera, programa)
    gerar_dataset_resnet(camera, programa)
    treinar_resnet(camera, programa, img_size)


def gerar_modelo_resnet50(camera, programa, img_size):
    limpar_dataset(camera, programa)
    gerar_dataset_resnet(camera, programa)
    treinar_resnet50(camera, programa, img_size)

#######################################resnet50 treinamento##############

def treinar_resnet50(camera, programa, img_size):

    dataset_dir = get_dataset_dir(camera, programa)

    # ✅ Augmentation no treino
    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ✅ Validação sem augmentation
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(dataset_dir / "train", transform=train_transform)
    val_dataset   = datasets.ImageFolder(dataset_dir / "val",   transform=val_transform)

    # ✅ Class weights para desbalanceamento
    class_counts  = torch.tensor(
        [train_dataset.targets.count(i) for i in range(2)], dtype=torch.float
    )
    class_weights = (1.0 / class_counts)
    class_weights = class_weights / class_weights.sum()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset, batch_size=16)

    model = models.resnet50(pretrained=True)

    # ✅ Congelar backbone, treinar só o topo primeiro
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 2)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)  # ✅ com peso

    # ✅ Fase 1 — treina só o fc (5 épocas, lr alto)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    scaler    = torch.amp.GradScaler("cuda")
    write_status({
        "running": True,
        "epoch": 0,
        "total_epochs": EPOCHS_RESNET,
        "progress": 0,
        "mensagem": "Iniciando treinamento ResNet..."
    })
    print("🔒 Fase 1 — treinando só o classificador...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        scheduler.step()
        print(f"  Epoch {epoch+1}/5 | Loss: {total_loss:.4f}")

    # ✅ Fase 2 — desbloqueia toda a rede com lr baixo
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # ✅ lr 10x menor
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)

    print("\n🔓 Fase 2 — fine-tuning completo...")
    for epoch in range(35):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        scheduler.step()
        print(f"  Epoch {epoch+1}/35 | Loss: {total_loss:.4f}")
        progress = int(((epoch + 1) / EPOCHS_RESNET) * 100)
        write_status({
            "running": True,
            "epoch": epoch + 1,
            "total_epochs": EPOCHS_RESNET,
            "progress": progress,
            "mensagem": f"Epoch {epoch + 1}/{EPOCHS_RESNET} | Loss: {total_loss:.4f}"
        })


    print(f"\nTrain size: {len(train_dataset)} | Val size: {len(val_dataset)}")

    from sklearn.metrics import classification_report, confusion_matrix
    labels_real, preds = avaliar_modelo(model, val_loader, device)
    print("\n📊 Relatório detalhado:")
    print(classification_report(labels_real, preds, target_names=["nok", "ok"]))
    print(confusion_matrix(labels_real, preds))

    torch.save(model.state_dict(), dataset_dir / "modelo_resnet50.pth")
    print(train_dataset.class_to_idx)
    print("🎯 Modelo ResNet treinado!")

from sklearn.metrics import accuracy_score

def avaliar_modelo(model, val_loader, device):

    model.eval()
    preds = []
    labels_real = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().numpy())
            labels_real.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_real, preds)
    

    print(f"\n🎯 Accuracy validação: {acc * 100:.2f}%")

    return labels_real, preds



### INFERÊNCIA COM O MODELO RESNET TREINADO (CLASSIFICAÇÃO DE ROIs) ###
import platform

def carregar_modelo(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    # 🚫 torch.compile NÃO no Windows
    if platform.system() != "Windows":
        model = torch.compile(model)

    return model, device
def carregar_modelo_resnet50(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights=None)

    # ✅ Mesma arquitetura usada no treino
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 2)
    )

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    if platform.system() != "Windows":
        model = torch.compile(model)

    return model, device


TRANSFORM_INFER = transforms.Compose([
    transforms.ToTensor(),  # aceita numpy HWC
    transforms.Resize((224, 224)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def classificar_roi(model, device, frame, roi):

    x, y, w, h = roi
    crop = frame[y:y+h, x:x+w]

    print("Shape crop:", crop.shape)
    print("Tipo:", crop.dtype)

    if crop.size == 0:
        return False

    # debug imagem original
    cv2.imwrite("teste_bgr.jpg", crop)

    # converter para RGB (modelo foi treinado em RGB)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # converter para PIL
    img_pil = Image.fromarray(crop)

    # debug RGB correto
    img_pil.save("teste_rgb.jpg")

    # aplicar transform igual ao treino
    img = TRANSFORM_INFER(img_pil).unsqueeze(0).to(device)

    # debug device
    print("Model device:", next(model.parameters()).device)
    print("Image device:", img.device)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    conf_val = conf.item()
    pred_val = pred.item()

    print("Pred:", pred_val, "Conf:", conf_val)

    # threshold de segurança
    if conf_val < 0.70:
        return False

    # classes do ImageFolder
    # 0 = nok
    # 1 = ok
    return pred_val == 1




    ################# mobilenetv2 (mais leve que a resnet18, pode ser mais rápido na inferência mas talvez menos preciso) #################


def gerar_dataset_mobilenetv2(camera, programa):

    config = configparser.ConfigParser()
    config.read(f'config_{camera}_{programa}.ini')

    n_rois = config.getint('Ferramentas', 'n_rois')

    dataset_dir = get_dataset_dir(camera, programa)

    for subset in ["train", "val"]:
        for classe in ["ok", "nok"]:
            (dataset_dir / subset / classe).mkdir(parents=True, exist_ok=True)

    amostras = []

    for i in range(1, n_rois + 1):
        sec = f'ROI{i}'
        if sec not in config:
            continue

        x = int(config[sec].get('x'))
        y = int(config[sec].get('y'))
        w = int(config[sec].get('width'))
        h = int(config[sec].get('height'))

        roi = (x, y, w, h)

        for classe in ["OK", "NOK"]:
            for idx in range(0, 100):
                nome = f'cam{camera}_ref_programa{programa}_{classe}_{idx if idx > 0 else ""}.png'
                caminho = os.path.join(IMG_DIR, nome)

                if os.path.exists(caminho):
                    amostras.append((caminho, roi, classe.lower()))

    random.shuffle(amostras)
    split = int(len(amostras) * TRAIN_SPLIT)

    for idx, (img_path, roi, classe) in enumerate(amostras):

        subset = "train" if idx < split else "val"

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, y, w, h = roi
        crop = img[y:y+h, x:x+w]

        if crop.size == 0:
            continue

        img_name = f"{Path(img_path).stem}_{idx}.jpg"
        out_path = dataset_dir / subset / classe / img_name

        Image.fromarray(crop).save(out_path)

    print("✅ Dataset MobileNetV2 gerado!")


from torchvision.models import mobilenet_v2

def treinar_mobilenet(camera, programa):

    dataset_dir = get_dataset_dir(camera, programa)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(dataset_dir / "train", transform=transform)
    val_dataset   = datasets.ImageFolder(dataset_dir / "val",   transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=16)

    # 🔥 MobileNetV2
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        model.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

    # salvar classes (evita inversão)
    torch.save(train_dataset.classes, dataset_dir / "classes.pth")

    # avaliação
    labels_real, preds = avaliar_modelo(model, val_loader, device)
    

    torch.save(model.state_dict(), dataset_dir / "modelo_mobilenetv2.pth")
    print("🎯 Modelo MobileNetV2 treinado!")


def gerar_modelo_mobilenet(camera, programa):
    limpar_dataset(camera, programa)
    gerar_dataset_mobilenetv2(camera, programa)
    treinar_mobilenet(camera, programa)

#################inferência com o modelo mobilenetv2 treinado #################

from torchvision.models import mobilenet_v2

def carregar_modelo_mobilenet(model_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    #class_names = torch.load(classes_path)

    return model, device


def classificar_roi_mobilenetv2(model, device, frame, roi):

    x, y, w, h = roi
    crop = frame[y:y+h, x:x+w]

    if crop.size == 0:
        return False

 
    cv2.imwrite("teste_nok.jpg", crop)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = transform(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs  = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    if conf.item() < 0.80: 
      return False  # tratar como NOK por segurança
    

    classe = "nok" if pred.item() == 0 else "ok"

    print("Classe:", classe, "Conf:", conf.item())

    if classe == "ok":
        status_rna = True
    elif classe == "nok":
        status_rna = False
    else:
        status_rna = True

    return status_rna
