# main.py
"""
Real-Time Interview Assistant - Main Loop
Video + Audio + Text + Drowsiness + Look-Away + Alarm
"""

import time
import numpy as np

from modules.video_module import get_video_features, release_video, draw_overlay
from modules.audio_module import get_audio_features
from modules.text_module import get_text_features
from modules.fusion_model import predict_result
from modules.drowsiness_module import update_sleep_status
from modules.intruder_module import detect_intruder

def main_loop():
    print("🚀 Starting AI Lie Detector (Enhanced Mode)...")

    try:
        while True:
            # 1️⃣ VIDEO FEATURES
            video_feat, frame = get_video_features()
            if frame is None:
                time.sleep(0.1)
                continue

            ear_value = float(video_feat[1])
            look_away_flag = float(video_feat[4])
            num_faces = int(video_feat[5])

            # 2️⃣ AUDIO FEATURES
            audio_feat, speaking_flag = get_audio_features()

            # 3️⃣ TEXT FEATURES
            text_feat = get_text_features()

            # 4️⃣ INTRUDER DETECTION
            intruder_detected = detect_intruder(num_faces)

            # 5️⃣ PREDICTION
            try:
                label, prob = predict_result(video_feat, audio_feat, text_feat)
            except Exception:
                label, prob = "Truth", 0.50

            # 6️⃣ DROWSINESS + CENTRAL ALARM SYSTEM
            drowsy = update_sleep_status(
                ear_value, 
                speaking_flag, 
                look_away_flag, 
                intruder_status=intruder_detected,
                lie_prob=prob
            )

            # 7️⃣ LOG OUTPUT
            ts = time.strftime("%H:%M:%S")
            print(
                f"[{ts}] P:{label}({prob:.2f}) | Ear:{ear_value:.2f} | "
                f"Away:{look_away_flag} | Speak:{speaking_flag} | Faces:{num_faces} | Intruder:{intruder_detected}"
            )

            # 8️⃣ OVERLAY WINDOW
            try:
                draw_overlay(frame, label, prob, drowsy, speaking_flag, look_away_flag, intruder_detected, num_faces)
            except Exception as e:
                print("Overlay error:", e)

            # Loop delay
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nUser interrupted - shutting down...")

    finally:
        release_video()


if __name__ == "__main__":
    main_loop()
