# placeholder
# modules/fusion_model.py
import numpy as np
import os
import pickle

MODEL_PATH = os.path.join('models', 'trained_model.pkl')

def _default_prob(video_feat, audio_feat, text_feat):
    drowsy_ear = float(video_feat[1]) if len(video_feat) > 1 else 0.3
    # Smaller EAR means more drowsy, so we use (1.0 - drowsy_ear)
    drowsiness_score = max(0.0, 1.0 - (drowsy_ear / 0.3)) 
    mean_audio_energy = float(np.mean(np.abs(audio_feat))) if audio_feat.size else 0.0
    hesitation = float(text_feat[1]) if text_feat.size > 1 else 0.0
    score = 0.3 * drowsiness_score + 0.2 * (mean_audio_energy / (1.0 + mean_audio_energy)) + 0.3 * (hesitation / (1.0 + hesitation))
    prob = float(max(0.0, min(1.0, score)))
    return prob

try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
    _HAS_MODEL = True
except Exception:
    model = None
    _HAS_MODEL = False

def predict_result(video_feat, audio_feat, text_feat):
    x = np.concatenate([np.ravel(video_feat), np.ravel(audio_feat), np.ravel(text_feat)])
    x = x.reshape(1, -1)
    if _HAS_MODEL and hasattr(model, "predict_proba"):
        try:
            prob = model.predict_proba(x)[0][1]
        except Exception:
            prob = _default_prob(video_feat, audio_feat, text_feat)
    else:
        prob = _default_prob(video_feat, audio_feat, text_feat)
    label = "Lie" if prob > 0.5 else "Truth"
    return label, float(prob)
