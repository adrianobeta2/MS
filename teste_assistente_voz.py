from TTS.api import TTS

tts = TTS("tts_models/pt_BR/cv/vits")

texto = "Olá! Agora estou usando uma voz muito mais natural e de alta qualidade."

tts.tts_to_file(text=texto, file_path="voz.wav")
