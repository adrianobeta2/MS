from pypylon import pylon
import threading
import cv2
import time

# Variáveis globais
last_frame_cam1 = None
last_frame_cam2 = None
lock_cam1 = threading.Lock()
lock_cam2 = threading.Lock()

# Configurações
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
SERIAL_NUMBER_CAM1 = "22746375"  # Substitua pelo serial real
SERIAL_NUMBER_CAM2 = "87654321"  # Substitua pelo serial real

def initialize_camera_basler(serial_number, cam_id):
    """ Inicializa uma câmera Basler e inicia a thread de captura """
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    
    camera = None
    for device in devices:
        if device.GetSerialNumber() == serial_number:
            camera = pylon.InstantCamera(tl_factory.CreateDevice(device))
            break

    if camera is None:
        print(f"Erro: Câmera {cam_id} não encontrada.")
        return None
    
    try:
        camera.Open()
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        threading.Thread(
            target=camera_loop_basler,
            args=(camera, converter, cam_id),
            daemon=True
        ).start()
        
        return camera
    except Exception as e:
        print(f"Erro ao iniciar câmera {cam_id}: {str(e)}")
        return None

def camera_loop_basler(camera, converter, cam_id):
    """ Captura frames continuamente com tratamento robusto de erros """
    global last_frame_cam1, last_frame_cam2
    
    while camera.IsGrabbing():
        grabResult = None
        try:
            # Timeout de 100ms (ajuste conforme necessário)
            grabResult = camera.RetrieveResult(100, pylon.TimeoutHandling_Return)
            
            if grabResult is None or not grabResult.GrabSucceeded():
                continue  # Pula para o próximo frame se falhar

            # Processamento do frame
            image = converter.Convert(grabResult)
            frame = image.GetArray()
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

            # Atualiza o frame correspondente
            if cam_id == 1:
                with lock_cam1:
                    last_frame_cam1 = frame
            elif cam_id == 2:
                with lock_cam2:
                    last_frame_cam2 = frame

        except Exception as e:
            print(f"Erro na câmera {cam_id}: {str(e)}")
        finally:
            if grabResult is not None:
                grabResult.Release()
        
        time.sleep(0.005)  # Pequeno delay para reduzir carga

# Inicialização das câmeras
camera1 = initialize_camera_basler(SERIAL_NUMBER_CAM1, 1)
#camera2 = initialize_camera_basler(SERIAL_NUMBER_CAM2, 2)

# Loop principal para exibição
try:
    while True:
        with lock_cam1:
            if last_frame_cam1 is not None:
                cv2.imshow("Câmera 1", last_frame_cam1)
        with lock_cam2:
            if last_frame_cam2 is not None:
                cv2.imshow("Câmera 2", last_frame_cam2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
    if camera1 is not None:
        camera1.StopGrabbing()
        camera1.Close()
    if camera2 is not None:
        camera2.StopGrabbing()
        camera2.Close()