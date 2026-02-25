from treinar_modelo_new import gerar_modelo, gerar_modelo_resnet50,gerar_dataset_resnet
import multiprocessing as mp

if __name__ == "__main__":
    mp.freeze_support()   # seguro no Windows
    gerar_modelo_resnet50(2, 1)
