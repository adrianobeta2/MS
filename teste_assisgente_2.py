from TTS.api import TTS
import torch

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

device = "cuda" if torch.cuda.is_available() else "cpu"
tts.to(device)

tts.tts_to_file(
    text="Olá, agora está funcionando corretamente.",
    file_path="voz.wav",
    language="pt"
)
