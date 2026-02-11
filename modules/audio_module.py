import sounddevice as sd
import numpy as np
import librosa

SR = 16000
DURATION = 1.0      # faster audio refresh
N_MFCC = 13

SPEAK_THRESHOLD = 0.02   # adjust later based on your mic


def get_audio_features():
    try:
        audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        if len(audio) < SR * 0.5:
            return np.zeros(N_MFCC), 0  # no sound

        # Compute MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
        feat = mfcc.mean(axis=1).astype(float)

        # Compute RMS energy for speaking detection
        rms = np.sqrt(np.mean(audio ** 2))

        speaking = 1 if rms > SPEAK_THRESHOLD else 0

        print("RMS:", rms, "| Speaking:", speaking)

        return feat, speaking

    except Exception:
        return np.zeros(N_MFCC), 0
