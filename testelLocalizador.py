import cv2
import numpy as np
import Localizador



template = cv2.imread('template.png', 0)
   

frame = cv2.imread('ref_programa1.png', 0)

    
roi_busca_template = (258, 103, 120, 124)  # (x, y, width, height)


roi_produto = (272, 290, 9, 9) 

ref_posTempl = (286, 116)  # (x, y) da posição de referência


nova_posicao =Localizador.ajustePosicao(template,roi_busca_template,frame,ref_posTempl,roi_produto)

print(f'Nova posicao: {nova_posicao}')

