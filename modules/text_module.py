# placeholder
# modules/text_module.py
import sounddevice as sd
import numpy as np
import json
from vosk import Model, KaldiRecognizer

SR = 16000
DURATION = 2.0

try:
    _VOSK_MODEL = Model('vosk-model-small-en-us-0.15')
    _RECOGNIZER = KaldiRecognizer(_VOSK_MODEL, SR)
except Exception:
    _VOSK_MODEL = None
    _RECOGNIZER = None

def get_text_features():
    if _RECOGNIZER is None:
        return np.array([0.0, 0.0], dtype=float)
    try:
        audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype='int16')
        sd.wait()
        data = audio.tobytes()
        if _RECOGNIZER.AcceptWaveform(data):
            res = json.loads(_RECOGNIZER.Result())
        else:
            res = json.loads(_RECOGNIZER.PartialResult())
        text = res.get("text", "")
        length_feature = float(len(text.split()))
        hesitation = float(text.count("um") + text.count("uh") + text.count("hmm"))
        return np.array([length_feature, hesitation], dtype=float)
    except Exception:
        return np.array([0.0, 0.0], dtype=float)
