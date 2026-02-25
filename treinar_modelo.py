
import configparser
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Pasta das imagens de referência
IMG_DIR = os.path.join(os.path.dirname(__file__), 'static', 'imagens')

def carregar_imagem(caminho, coordenadas):
    """Carrega imagem em grayscale, recorta ROI e redimensiona"""
    x, y, w, h = coordenadas
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    roi = img[y:y+h, x:x+w]
    roi = cv2.resize(roi, (28,28))
    roi = roi.astype("float32") / 255.0
    return roi.reshape(28,28,1)

def treinar(camera, programa):
    x, y = [], []
    

    config = configparser.ConfigParser()
    config.read(f'config_{camera}_{programa}.ini')
    n_rois = config.getint('Ferramentas', 'n_rois')
  


    # Listar imagens OK e NOK
    template_images_ok = [
        f'cam{camera}_ref_programa{programa}_OK.png',
        f'cam{camera}_ref_programa{programa}_OK_1.png',
        f'cam{camera}_ref_programa{programa}_OK_2.png',
        f'cam{camera}_ref_programa{programa}_OK_3.png',
        f'cam{camera}_ref_programa{programa}_OK_4.png',
        f'cam{camera}_ref_programa{programa}_OK_5.png',
        f'cam{camera}_ref_programa{programa}_OK_6.png',
        f'cam{camera}_ref_programa{programa}_OK_7.png',
        f'cam{camera}_ref_programa{programa}_OK_8.png',
        f'cam{camera}_ref_programa{programa}_OK_9.png',
        f'cam{camera}_ref_programa{programa}_OK_10.png',
        f'cam{camera}_ref_programa{programa}_OK_11.png',
        f'cam{camera}_ref_programa{programa}_OK_12.png',
        f'cam{camera}_ref_programa{programa}_OK_13.png',
        f'cam{camera}_ref_programa{programa}_OK_14.png',
        f'cam{camera}_ref_programa{programa}_OK_15.png',
        f'cam{camera}_ref_programa{programa}_OK_16.png',
        f'cam{camera}_ref_programa{programa}_OK_17.png',
        f'cam{camera}_ref_programa{programa}_OK_18.png',
        f'cam{camera}_ref_programa{programa}_OK_19.png',
        f'cam{camera}_ref_programa{programa}_OK_20.png',
        f'cam{camera}_ref_programa{programa}_OK_21.png',
        f'cam{camera}_ref_programa{programa}_OK_22.png',
        f'cam{camera}_ref_programa{programa}_OK_23.png',
        f'cam{camera}_ref_programa{programa}_OK_24.png',
        f'cam{camera}_ref_programa{programa}_OK_25.png',
        f'cam{camera}_ref_programa{programa}_OK_26.png',
        f'cam{camera}_ref_programa{programa}_OK_27.png',
        f'cam{camera}_ref_programa{programa}_OK_28.png',
        f'cam{camera}_ref_programa{programa}_OK_29.png',
        f'cam{camera}_ref_programa{programa}_OK_30.png',
        f'cam{camera}_ref_programa{programa}_OK_31.png',
    ]

    template_images_nok = [
        f'cam{camera}_ref_programa{programa}_NOK.png',
        f'cam{camera}_ref_programa{programa}_NOK_1.png',
        f'cam{camera}_ref_programa{programa}_NOK_2.png',
        f'cam{camera}_ref_programa{programa}_NOK_3.png',
        f'cam{camera}_ref_programa{programa}_NOK_4.png',
        f'cam{camera}_ref_programa{programa}_NOK_5.png',
        f'cam{camera}_ref_programa{programa}_NOK_6.png',
        f'cam{camera}_ref_programa{programa}_NOK_7.png',
        f'cam{camera}_ref_programa{programa}_NOK_8.png',
        f'cam{camera}_ref_programa{programa}_NOK_9.png',
        f'cam{camera}_ref_programa{programa}_NOK_10.png',
        f'cam{camera}_ref_programa{programa}_NOK_11.png',
        f'cam{camera}_ref_programa{programa}_NOK_12.png',
        f'cam{camera}_ref_programa{programa}_NOK_13.png',
        f'cam{camera}_ref_programa{programa}_NOK_14.png',
        f'cam{camera}_ref_programa{programa}_NOK_15.png',
        f'cam{camera}_ref_programa{programa}_NOK_16.png',
        f'cam{camera}_ref_programa{programa}_NOK_17.png',
        f'cam{camera}_ref_programa{programa}_NOK_18.png',
        f'cam{camera}_ref_programa{programa}_NOK_19.png',
        f'cam{camera}_ref_programa{programa}_NOK_20.png',
        f'cam{camera}_ref_programa{programa}_NOK_21.png',
        f'cam{camera}_ref_programa{programa}_NOK_22.png',
        f'cam{camera}_ref_programa{programa}_NOK_23.png',
        f'cam{camera}_ref_programa{programa}_NOK_24.png',
        f'cam{camera}_ref_programa{programa}_NOK_25.png',
        f'cam{camera}_ref_programa{programa}_NOK_26.png',
        f'cam{camera}_ref_programa{programa}_NOK_27.png',
        f'cam{camera}_ref_programa{programa}_NOK_28.png',
        f'cam{camera}_ref_programa{programa}_NOK_29.png',
        f'cam{camera}_ref_programa{programa}_NOK_30.png',
        f'cam{camera}_ref_programa{programa}_NOK_31.png',

    ]
    
    for i in range(1, n_rois+1):  # Três ROIs
            section_name = f'ROI{i}'
            if section_name in config:
                x_coord = int(config[section_name]['x'])
                if(x_coord == None):
                     x_coord = int(config[section_name]['x_anterior'])

                y_coord = int(config[section_name]['y'])
                if(y_coord == None):
                     y_coord = int(config[section_name]['y_anterior'])

                w_cood = int(config[section_name]['width'])
                if(w_cood == None):
                     w_cood = int(config[section_name]['width_anterior'])
                     
                h_coord = int(config[section_name]['height'])
                if(h_coord == None):
                     h_coord = int(config[section_name]['height_anterior'])
            coordenadas = (x_coord,y_coord,w_cood,h_coord)

            # 🔹 Reinicializar as listas a cada ROI
            x, y = [], []
            # Carregar imagens
            for fname in template_images_ok:
                caminho = os.path.join(IMG_DIR, fname)
                if os.path.exists(caminho):
                    x.append(carregar_imagem(caminho, coordenadas))
                    y.append(1)

            for fname in template_images_nok:
                caminho = os.path.join(IMG_DIR, fname)
                if os.path.exists(caminho):
                    x.append(carregar_imagem(caminho, coordenadas))
                    y.append(0)

            x = np.array(x)
            y = np.array(y)

            # Data augmentation
            datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1
            )

            # Modelo CNN pequeno
            model = Sequential([
                Input(shape=(28,28,1)),
                Conv2D(16, (3,3), activation='relu', padding='same'),
                MaxPooling2D((2,2)),
             # 🔹 Segunda camada convolucional leve
                Conv2D(32, (3,3), activation='relu', padding='same'),
                MaxPooling2D((2,2)),
             # 🔹 Terceira camada convolucional leve
                Conv2D(64, (3,3), activation='relu', padding='same'),
                MaxPooling2D((2,2)),
            
                Flatten(),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            

            # Treinar com data augmentation
            model.fit(datagen.flow(x, y, batch_size=2), epochs=20)

            # Salvar modelo
            modelo_path = f"modelo_cam{camera}_prog{programa}_{section_name}.keras"
            model.save(modelo_path)
            print(f"Modelo salvo em '{modelo_path}'")

    return model.summary()

def testar_modelo(model, frame, coordenadas):
    """Testa modelo em um frame capturado"""
    img = carregar_imagem(frame, coordenadas).reshape(1,28,28,1)
    pred = model.predict(img)
    resultado = 1 if pred[0][0] >= 0.5 else 0
    return resultado


   