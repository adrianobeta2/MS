import numpy as np
import soundfile as sf

samplerate = 44100
duration = 0.25
frequency = 2800  # parecido com o beep do TikTok

t = np.linspace(0, duration, int(samplerate * duration), False)
tone = 0.5 * np.sin(2 * np.pi * frequency * t)

# fade in / fade out
fade_len = int(0.02 * samplerate)
fade = np.linspace(0, 1, fade_len)
tone[:fade_len] *= fade
tone[-fade_len:] *= fade[::-1]

sf.write("beep_tiktok_style.wav", tone, samplerate)
