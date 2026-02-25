
import cv2
from  treinar_modelo_new import carregar_modelo, classificar_roi

frame = cv2.imread("dataset/cam2_prog1/val/ok/cam2_ref_programa1_OK_3_88.jpg")
h, w, _ = frame.shape
roi = (0, 0, w, h)


model, device = carregar_modelo("dataset/cam2_prog1/modelo_resnet34.pth")

resultado = classificar_roi(model, device, frame, roi)
print(resultado)