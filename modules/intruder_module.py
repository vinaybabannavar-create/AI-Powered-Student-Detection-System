# modules/intruder_module.py
"""
Intruder Detection Module
Detects when additional people enter the room and triggers alarm
Useful for detecting thieves or unauthorized entry
"""

import time
import threading
from modules.drowsiness_module import play_alarm_sound

# Configuration
BASELINE_DURATION = 3.0      # Seconds to establish baseline person count
INTRUDER_DURATION = 1.5      # Seconds of increased count before alarm
ALARM_COOLDOWN = 10.0        # Seconds between intruder alarms
RESET_TIMEOUT = 10.0         # Seconds of empty room to reset baseline

# State variables
baseline_person_count = None
baseline_start_time = None
intruder_start_time = None
last_intruder_alarm = 0
last_person_count = 0
last_seen_time = time.time()

# Tracking
person_count_history = []

def play_intruder_alarm():
    """Trigger the centralized intruder alarm."""
    play_alarm_sound("Intruder / Security Alert")

def detect_intruder(num_faces):
    global baseline_person_count, baseline_start_time, intruder_start_time
    global last_intruder_alarm, last_person_count, last_seen_time, person_count_history
    
    now = time.time()
    intruder_detected = False
    
    # -------------------------------------------------------
    # 1️⃣ BASELINE ESTABLISHMENT
    # -------------------------------------------------------
    if baseline_person_count is None:
        if baseline_start_time is None:
            baseline_start_time = now
            person_count_history = []
            print("🔍 Establishing baseline person count. Please stay in frame alone...")
        
        person_count_history.append(num_faces)
        
        if now - baseline_start_time >= BASELINE_DURATION:
            if person_count_history:
                baseline_person_count = max(set(person_count_history), 
                                           key=person_count_history.count)
                print(f"✅ Baseline set: {baseline_person_count} person(s). Security ACTIVE.")
            else:
                baseline_person_count = 0
        return False
    
    # -------------------------------------------------------
    # 2️⃣ RESET BASELINE IF ROOM EMPTY
    # -------------------------------------------------------
    if num_faces == 0:
        if now - last_seen_time >= RESET_TIMEOUT:
            print("🔄 Room empty - resetting security baseline...")
            baseline_person_count = None
            baseline_start_time = None
            intruder_start_time = None
            person_count_history = []
        return False
    else:
        last_seen_time = now
    
    # -------------------------------------------------------
    # 3️⃣ INTRUDER DETECTION
    # -------------------------------------------------------
    if num_faces > baseline_person_count:
        if intruder_start_time is None:
            intruder_start_time = now
            print(f"⚠️ Warning: Person count increased to {num_faces}!")
        
        elif now - intruder_start_time >= INTRUDER_DURATION:
            if now - last_intruder_alarm >= ALARM_COOLDOWN:
                print(f"🚨 ALARM: INTRUDER DETECTED! (Count: {num_faces})")
                play_intruder_alarm()
                last_intruder_alarm = now
                intruder_detected = True
    else:
        if intruder_start_time is not None:
            print(f"✓ Person count back to baseline: {num_faces}")
        intruder_start_time = None
    
    last_person_count = num_faces
    return intruder_detected


def get_intruder_status():
    """
    Get current intruder detection status for display.
    
    Returns:
        dict with status information
    """
    return {
        'baseline': baseline_person_count,
        'establishing': baseline_person_count is None,
        'last_count': last_person_count
    }
