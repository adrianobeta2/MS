from pypylon import pylon

tl_factory = pylon.TlFactory.GetInstance()
devices = tl_factory.EnumerateDevices()

print("Câmeras encontradas:")
for d in devices:
    print("Classe:", d.GetDeviceClass())
    print("UserDefinedName:", d.GetUserDefinedName())
    print("IP:", d.GetIpAddress())
    print("-" * 30)



from pypylon import pylon
import cv2

# ID DA CÂMERA (igual ao Pylon Viewer)
CAMERA_NAME = "Cam-01"

# Factory
tl_factory = pylon.TlFactory.GetInstance()

# Descobrir dispositivos
devices = tl_factory.EnumerateDevices()
device = None

for d in devices:
    print("Encontrado:", d.GetUserDefinedName(), d.GetDeviceClass())
    if d.GetUserDefinedName() == CAMERA_NAME:
        device = d
        break

if device is None:
    raise RuntimeError("Câmera não encontrada")

print("Abrindo câmera...")

# Criar câmera
camera = pylon.InstantCamera(tl_factory.CreateDevice(device))
camera.Open()

# ===== CONFIGURAÇÃO MÍNIMA E SEGURA =====
camera.AcquisitionMode.Value = "Continuous"

# DESLIGAR TRIGGER (CRÍTICO)
try:
    camera.TriggerSelector.Value = "FrameStart"
    camera.TriggerMode.Value = "Off"
except:
    pass

# Converter para OpenCV
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

print("Iniciando aquisição...")
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

print("Pressione ESC para sair")

while camera.IsGrabbing():
    try:
        grab = camera.RetrieveResult(
            3000, pylon.TimeoutHandling_ThrowException
        )

        if grab.GrabSucceeded():
            image = converter.Convert(grab)
            frame = image.GetArray()

            cv2.imshow("Basler GigE Teste", frame)

        grab.Release()

        if cv2.waitKey(1) == 27:
            break

    except Exception as e:
        print("Erro de captura:", e)
        break

print("Encerrando...")

camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()