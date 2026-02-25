import cv2
import numpy as np

def _to_gray(img):
    if img is None:
        return None
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def _scale_keep_aspect(img, max_side=600):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def match_orb_robust(template_img, probe_img,
                     min_inliers=10,
                     ratio_test=0.75,
                     try_fallback=True,
                     debug=False):
    """
    Tenta encontrar o template na imagem probe usando ORB (com fallback para AKAZE/BRISK).
    Retorna (found:bool, debug_image, info:dict)
    info contém n_kp_template, n_kp_probe, n_good_matches, n_inliers, used_detector.
    """

    # PREPROCESS
    t = _to_gray(template_img)
    q = _to_gray(probe_img)
    if t is None or q is None:
        raise ValueError("Template ou probe inválido (None).")

    # equaliza contraste (ajuda bastante com iluminação variada)
    t = cv2.equalizeHist(t)
    q = cv2.equalizeHist(q)

    # padroniza escala (opcional) — ajusta se as imagens forem muito grandes
    t = _scale_keep_aspect(t, max_side=400)
    q = _scale_keep_aspect(q, max_side=800)

    detectors = []
    # principal: ORB (binário)
    detectors.append(("ORB", cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, edgeThreshold=15, patchSize=31, fastThreshold=20)))
    if try_fallback:
        # AKAZE costuma funcionar bem em texturas fracas
        if hasattr(cv2, 'AKAZE_create'):
            detectors.append(("AKAZE", cv2.AKAZE_create()))
        # BRISK como alternativa (binário)
        detectors.append(("BRISK", cv2.BRISK_create()))

    best_result = {
        "found": False,
        "debug_img": None,
        "info": {
            "n_kp_template": 0,
            "n_kp_probe": 0,
            "n_good_matches": 0,
            "n_inliers": 0,
            "used_detector": None
        }
    }

    for name, detector in detectors:
        kp1, des1 = detector.detectAndCompute(t, None)
        kp2, des2 = detector.detectAndCompute(q, None)

        n_kp1 = 0 if kp1 is None else len(kp1)
        n_kp2 = 0 if kp2 is None else len(kp2)

        if des1 is None or des2 is None or n_kp1 == 0 or n_kp2 == 0:
            if debug:
                print(f"[{name}] Sem descritores detectados. kp_template={n_kp1}, kp_probe={n_kp2}")
            continue

        # escolher norma dependendo do tipo do descritor (binário -> HAMMING)
        norm = cv2.NORM_HAMMING if des1.dtype == np.uint8 else cv2.NORM_L2
        bf = cv2.BFMatcher(norm, crossCheck=False)

        # knnMatch e ratio test (Lowe)
        knn_matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m_n in knn_matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < ratio_test * n.distance:
                good.append(m)

        n_good = len(good)
        if debug:
            print(f"[{name}] kp1={n_kp1}, kp2={n_kp2}, good_matches={n_good}")

        # Necessário pelo menos 4 pontos para homografia
        n_inliers = 0
        mask = None
        if n_good >= 4:
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
            if mask is not None:
                n_inliers = int(mask.sum())
            if debug:
                print(f"[{name}] inliers={n_inliers}")
        else:
            if debug:
                print(f"[{name}] matches insuficientes para homografia (need>=4, got {n_good})")

        # critério: homografia com inliers suficientes ou muitos good matches
        found = False
        if n_inliers >= min_inliers:
            found = True
        elif n_good >= max(min_inliers, 20):  # fallback: muitos matches sem homografia
            found = True

        # desenhar matches (aplica máscara de inliers se disponível)
        matches_to_draw = good[:100]  # desenhar no máximo 100
        if mask is not None:
            # convert mask para lista compatível com drawMatches (correspondente aos matches selecionados)
            mask_list = mask.ravel().tolist()
            # expand/cortar para o tanto de matches_to_draw
            draw_mask = mask_list[:len(matches_to_draw)]
        else:
            draw_mask = None

        debug_img = cv2.drawMatches(t, kp1, q, kp2, matches_to_draw, None,
                                   matchColor=(0,255,0), singlePointColor=(255,0,0),
                                   matchesMask=draw_mask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        best_result["debug_img"] = debug_img
        best_result["info"].update({
            "n_kp_template": n_kp1,
            "n_kp_probe": n_kp2,
            "n_good_matches": n_good,
            "n_inliers": n_inliers,
            "used_detector": name
        })
        best_result["found"] = found

        # se achou com esse detector, retorna imediatamente
        if found:
            if debug:
                print(f"FOUND with {name}: inliers={n_inliers}, good={n_good}")
            return best_result["found"], best_result["debug_img"], best_result["info"]

        # caso não encontrado, tenta próximo detector (fallback)
        if debug:
            print(f"NOT found with {name}; tentando próximo detector (se houver).")

    # nenhum detector encontrou correspondência suficiente
    return best_result["found"], best_result["debug_img"], best_result["info"]

template = cv2.imread("templateOK.png", cv2.IMREAD_GRAYSCALE)
frame = cv2.imread("teste2.png", cv2.IMREAD_GRAYSCALE)

found, dbg_img, info = match_orb_robust(template, frame, min_inliers=30, debug=True)
print(info)
if dbg_img is not None:
    cv2.imwrite("debug_matches.png", dbg_img)
if found:
    print("Peça detectada ✅")
else:
    print("Peça NÃO detectada ❌")
