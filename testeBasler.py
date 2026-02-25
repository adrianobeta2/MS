from pypylon import pylon
import cv2

# Obtém a lista de câmeras disponíveis
tl_factory = pylon.TlFactory.GetInstance()
devices = tl_factory.EnumerateDevices()

# Exibe os IDs das câmeras disponíveis
for i, device in enumerate(devices):
    print(f"Câmera {i}: {device.GetFriendlyName()} - Serial: {device.GetSerialNumber()}")

# Defina o Serial Number da câmera desejada
serial_number = "22069800"  # Altere para o número de série da sua câmera

# Encontra a câmera com o serial especificado
camera = None
for device in devices:
    if device.GetSerialNumber() == serial_number:
        camera = pylon.InstantCamera(tl_factory.CreateDevice(device))
        break

if camera is None:
    print("Erro: Câmera com o Serial Number especificado não encontrada.")
    exit()

# Inicia a captura de vídeo
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

# Configura a conversão para o formato OpenCV
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Converte a imagem para OpenCV
        image = converter.Convert(grabResult)
        img = image.GetArray()

        cv2.imshow('Basler Camera', img)
        if cv2.waitKey(1) == 27:  # Pressione ESC para sair
            break

    grabResult.Release()

# Libera recursos
camera.StopGrabbing()
cv2.destroyAllWindows()
