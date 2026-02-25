
import torch
print("Torch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())
print("Qtde GPUs:", torch.cuda.device_count())
print("GPU:", torch.cuda.get_device_name(0))


