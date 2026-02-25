import cv2
import numpy as np



def ajustePosicao(template, roi_busca_templ,frame,ref_posXY_templ,roi_produto):

    # Carregar a imagem modelo (template) da peça
    #template = cv2.imread('template.png', 0)

    if len(template.shape) > 2:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    if len(frame.shape) > 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    w, h = template.shape[::-1]

    # Capturar imagem da câmera (ou carregar uma imagem de exemplo)
    #frame = cv2.imread('ref_programa1.png', 0)

    # Definir a ROI para o template matching (região onde a peça será procurada)
    #roi_template = (258, 103, 120, 124)  # (x, y, width, height)
    x_roi, y_roi, w_roi, h_roi = roi_busca_templ

    # Extrair a região da ROI do frame
    roi = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]



    # Aplicar template matching para encontrar a peça
    res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Ajustar a posição para a imagem original
    top_left = (max_loc[0] + x_roi, max_loc[1] + y_roi)
    bottom_right = (top_left[0] + w, top_left[1] + h)


    # Ajustar a ROI com base no deslocamento da peça
    roi_template = (
        top_left[0],  # Novo x
        top_left[1],  # Novo y
        bottom_right[0],            # Largura (não muda)
        bottom_right[1]             # Altura (não muda)
    )


    # Posição de referência da peça (exemplo: posição inicial)
    #ref_pos = (286, 116)  # (x, y) da posição de referência

    # Posição atual da peça
    current_pos = top_left  # (x, y) da posição detectada

    # Calcular o deslocamento (Δx, Δy)
    delta_x = current_pos[0] - ref_posXY_templ[0]
    delta_y = current_pos[1] - ref_posXY_templ[1]

    # Coordenadas originais da ROI (exemplo)
     #roi_original = (272, 290, 9, 9)  # (x, y, width, height)

    # Ajustar a ROI com base no deslocamento da peça
    roi_ajustada = (
        roi_produto[0] + delta_x,  # Novo x
        roi_produto[1] + delta_y,  # Novo y
        roi_produto[2],            # Largura (não muda)
        roi_produto[3]             # Altura (não muda)
    )

    
   

    return roi_ajustada, current_pos


def coord_ref(template, roi_busca_templ,frame):
    
    # Verificar se o template e o frame estão em escala de cinza
    if len(template.shape) > 2:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    if len(frame.shape) > 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #template = cv2.imread('template.png', 0)
    w, h = template.shape[::-1]

    # Capturar imagem da câmera (ou carregar uma imagem de exemplo)
    #frame = cv2.imread('ref_programa1.png', 0)

    # Definir a ROI para o template matching (região onde a peça será procurada)
    #roi_template = (258, 103, 120, 124)  # (x, y, width, height)

    
    x_roi, y_roi, w_roi, h_roi = roi_busca_templ

    # Extrair a região da ROI do frame
    roi = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]



    # Aplicar template matching para encontrar a peça
    res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Ajustar a posição para a imagem original
    top_left = (max_loc[0] + x_roi, max_loc[1] + y_roi)
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Posição de referência da peça (exemplo: posição inicial)
    #ref_pos = (286, 116)  # (x, y) da posição de referência

    # Posição atual da peça
    posicao_ref = top_left  # (x, y) da posição detectada

    
    
   

    return posicao_ref