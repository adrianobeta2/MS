import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import face_recognition

print("Versões:")
import dlib, numpy
print("dlib:", dlib.__version__)
print("numpy:", numpy.__version__)

img = cv2.imread("teste.jpeg")  # use uma imagem real com rosto
if img is None:
    raise RuntimeError("Imagem teste.jpg não encontrada")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

print("Detectando rostos...")
locs = face_recognition.face_locations(
    small,
    model="hog",
    number_of_times_to_upsample=0
)

print("Rostos detectados:", locs)
