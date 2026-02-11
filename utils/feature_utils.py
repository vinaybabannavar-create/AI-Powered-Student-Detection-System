# placeholder
# utils/feature_utils.py
import numpy as np

def safe_concat(*arrays):
    out = []
    for a in arrays:
        if a is None:
            continue
        out.append(np.ravel(a))
    if not out:
        return np.array([])
    return np.concatenate(out)
