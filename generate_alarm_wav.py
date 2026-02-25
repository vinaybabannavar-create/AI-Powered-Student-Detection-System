# generate_alarm_wav.py
# Generates a short alarm sound (alarm.mp3) into /alarm (requires soundfile)
import os
import numpy as np
import soundfile as sf

os.makedirs('alarm', exist_ok=True)
wav_path = os.path.join('alarm', 'alarm.wav')
mp3_path = os.path.join('alarm', 'alarm.mp3')

sr = 22050
t = np.linspace(0, 1.5, int(sr*1.5), endpoint=False)
carrier = 0.6 * np.sin(2 * np.pi * 1000 * t)
mod = 0.5 * (1 + np.sign(np.sin(2 * np.pi * 5 * t)))
alarm = carrier * mod
alarm = alarm / (np.max(np.abs(alarm)) + 1e-9)

sf.write(wav_path, alarm, sr)
# If playsound supports wav on your system, you can use wav directly.
# If you prefer mp3, convert outside Python (ffmpeg) or rename if .wav works.
print("Alarm saved to:", wav_path)
