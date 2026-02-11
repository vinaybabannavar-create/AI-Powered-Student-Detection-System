# modules/drowsiness_module.py
import time
import threading
import os
# import simpleaudio as sa (Removed for cloud compatibility)

# Thresholds
EAR_THRESHOLD = 0.20        # Eye aspect ratio threshold (calibrated for MediaPipe)
SLEEP_SECONDS = 2.5         # Eyes closed for 2.5 sec → ALARM
LOOK_AWAY_SECONDS = 3.0     # Looking away for 3 sec → ALARM
ALARM_COOLDOWN = 10.0       # Cooldown between alarms

# Global State
sleep_start = None
look_away_start = None
last_alarm_time = 0
alarm_playing = False

def play_alarm_sound(reason):
    """
    Server-side alarm is disabled for cloud deployment.
    (Audio should be handled on client-side if needed)
    """
    print(f">>> [ALARM LOG] Triggered: {reason}")
    return

def update_sleep_status(ear_value, speaking_flag, look_away_flag, intruder_status=False, lie_prob=0.0):
    """
    Central Alarm Manager.
    Returns drowsy boolean for UI display.
    """
    global sleep_start, look_away_start
    now = time.time()
    is_drowsy = False

    # 1. Drowsiness (EAR)
    if ear_value < EAR_THRESHOLD:
        if sleep_start is None:
            sleep_start = now
        elif now - sleep_start >= SLEEP_SECONDS:
            is_drowsy = True
            play_alarm_sound("Drowsiness / Eyes Closed")
    else:
        sleep_start = None

    # 2. Look Away
    if look_away_flag == 1:
        if look_away_start is None:
            look_away_start = now
        elif now - look_away_start >= LOOK_AWAY_SECONDS:
            play_alarm_sound("Attention Loss (Looking Away)")
    else:
        look_away_start = None

    # 3. Intruder / Multiple People
    if intruder_status:
        play_alarm_sound("Intruder / Security Alert")

    # 4. High Lie Probability
    if lie_prob > 0.8:
        play_alarm_sound("High Deception Detected")

    return is_drowsy
