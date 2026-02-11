import cv2
import numpy as np
import time
import threading
import mediapipe as mp

# MediaPipe Initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,  # Only one for detail analysis (closest)
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,  # 1 is for long-range (up to 5m+)
    min_detection_confidence=0.35 # Lowered for better sensitivity in diverse lighting
)

# Hybrid: OpenCV HOG People Detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Camera Global
cap = None
is_fullscreen = False
window_name = "AI Lie Detector - Live Feed"

def init_camera():
    """Initialize webcam with optimized settings."""
    global cap
    if cap is None:
        # Use DirectShow on Windows for faster startup and stability
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Create resizable window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)  # Standard larger start size
        
        time.sleep(1.5) # Increased for stability
        print("🎥 Webcam Initialized with MediaPipe (Resizable)")

def calculate_ear(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio (EAR)."""
    try:
        # Vertical distances
        p1 = np.array(landmarks[eye_indices[1]])
        p6 = np.array(landmarks[eye_indices[5]])
        p2 = np.array(landmarks[eye_indices[2]])
        p5 = np.array(landmarks[eye_indices[4]])
        v1 = np.linalg.norm(p1 - p6)
        v2 = np.linalg.norm(p2 - p5)
        # Horizontal distance
        p0 = np.array(landmarks[eye_indices[0]])
        p3 = np.array(landmarks[eye_indices[3]])
        h = np.linalg.norm(p0 - p3)
        ear = (v1 + v2) / (2.0 * h)
        return ear
    except:
        return 0.0

def get_head_pose(landmarks, image_shape):
    """Estimate head pose (Yaw) to detect looking away."""
    rows, cols, _ = image_shape
    # Nose tip, Chin, Left Eye, Right Eye, Left Mouth, Right Mouth
    # MediaPipe indices: 1, 152, 33, 263, 61, 291
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye corner
        (225.0, 170.0, -135.0),      # Right eye corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    image_points = np.array([
        (landmarks[1][0] * cols, landmarks[1][1] * rows),
        (landmarks[152][0] * cols, landmarks[152][1] * rows),
        (landmarks[33][0] * cols, landmarks[33][1] * rows),
        (landmarks[263][0] * cols, landmarks[263][1] * rows),
        (landmarks[61][0] * cols, landmarks[61][1] * rows),
        (landmarks[291][0] * cols, landmarks[291][1] * rows)
    ], dtype="double")

    focal_length = cols
    center = (cols / 2, rows / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    rmat, _ = cv2.Rodrigues(rotation_vector)
    # decomposeProjectionMatrix returns a tuple of 7 elements
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(np.hstack((rmat, translation_vector)))
    yaw = euler_angles[1][0]
    return abs(yaw)

def process_image(frame):
    """Headless processing of a single frame (no camera required)."""
    if frame is None:
        return np.zeros(10, float), None

    # 1. FACE DETECTION (High Accuracy)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detect_results = face_detection.process(rgb_frame)
    num_faces = 0
    if detect_results.detections:
        num_faces = len(detect_results.detections)
        for detection in detect_results.detections:
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # 2. BODY DETECTION (HOG)
    bodies, _ = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    num_bodies = len(bodies)
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    final_count = max(num_faces, num_bodies)

    features = np.zeros(10, float)
    features[5] = float(final_count)
    return features, frame

def get_video_features():
    init_camera()
    ret, frame = cap.read()
    if not ret:
        return np.zeros(10, float), None

    # 1. FACE DETECTION (High Accuracy)
    # We use full resolution here for better distance detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detect_results = face_detection.process(rgb_frame)
    num_faces = 0
    if detect_results.detections:
        num_faces = len(detect_results.detections)
        for detection in detect_results.detections:
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
            # Drawing Cyan boxes for faces
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # 2. BODY DETECTION (HOG) - Detects people from back/side
    # Using original frame for bodies to get more detail
    bodies, _ = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    num_bodies = len(bodies)
    for (x, y, w, h) in bodies:
        # Drawing Yellow boxes for bodies
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # FINAL COUNT: Max of faces or bodies
    final_count = max(num_faces, num_bodies)

    # 3. DETAIL ANALYSIS (FaceMesh) for closest person
    # Resize only for mesh to save CPU
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_small)
    avg_ear = 0.30
    look_away_flag = 0.0
    
    if results.multi_face_landmarks:
        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            landmarks.append((lm.x, lm.y, lm.z))

        left_eye = [362, 385, 387, 263, 373, 380]
        right_eye = [33, 160, 158, 133, 153, 144]
        
        ear_l = calculate_ear(landmarks, left_eye)
        ear_r = calculate_ear(landmarks, right_eye)
        avg_ear = (ear_l + ear_r) / 2.0
        
        try:
            yaw = get_head_pose(landmarks, small_frame.shape)
            look_away_flag = 1.0 if yaw > 20 else 0.0
        except: pass

        # Draw landmarks for the primary user
        for lm in results.multi_face_landmarks[0].landmark:
            ih, iw, _ = frame.shape
            cv2.circle(frame, (int(lm.x * iw), int(lm.y * ih)), 1, (0, 255, 0), -1)

    features = np.zeros(10, float)
    features[1] = avg_ear
    features[4] = look_away_flag
    features[5] = float(final_count)
    return features, frame

def draw_overlay(frame, label, prob, drowsy, speaking_flag, look_away_flag, intruder_status=False, num_faces=0):
    if frame is None: return

    # Semi-transparent background for text area
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 250), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    color = (0, 255, 0) if label == "Truth" else (0, 0, 255)
    
    cv2.putText(frame, f"Analysis: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Conf: {prob:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if drowsy:
        cv2.putText(frame, "!!! DROWSY !!!", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    if look_away_flag:
        cv2.putText(frame, "LOOK AWAY ALERT", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if speaking_flag:
        cv2.putText(frame, "SPEAKING", (10, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(frame, f"Student Count: {num_faces}", (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    
    if intruder_status:
        cv2.putText(frame, "!!! INTRUDER !!!", (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # cv2.imshow(window_name, frame)
    
    global is_fullscreen
    key = cv2.waitKey(1) & 0xFF
    
    # Toggle Fullscreen with 'f'
    if key == ord('f'):
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            
    # Exit if 'q' or 'Esc' is pressed
    if key == ord('q') or key == 27:
        release_video()
        import sys
        sys.exit()

def release_video():
    global cap
    if cap is not None:
        cap.release()
        cv2.destroyAllWindows()
        print("🎥 Camera Released")
