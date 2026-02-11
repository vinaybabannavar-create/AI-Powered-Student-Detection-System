# placeholder
# utils/preprocessing.py
def normalize_audio(a):
    import numpy as np
    a = a.astype(float)
    mx = np.max(np.abs(a)) if a.size else 0.0
    return a / (mx + 1e-9) if mx > 0 else a
