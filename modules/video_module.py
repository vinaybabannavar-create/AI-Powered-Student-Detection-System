import cv2
import numpy as np
import time
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---- Model paths ----
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FACE_DETECTOR_MODEL = os.path.join(BASE_DIR, 'models', 'mediapipe', 'blaze_face_short_range.tflite')
FACE_LANDMARKER_MODEL = os.path.join(BASE_DIR, 'models', 'mediapipe', 'face_landmarker.task')

# ---- MediaPipe Tasks API Initialization ----
# Face Detector
_face_detector_options = vision.FaceDetectorOptions(
    base_options=mp_python.BaseOptions(model_asset_path=FACE_DETECTOR_MODEL),
    min_detection_confidence=0.35
)
face_detector = vision.FaceDetector.create_from_options(_face_detector_options)

# Face Landmarker (replaces FaceMesh)
_face_landmarker_options = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL),
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False
)
face_landmarker = vision.FaceLandmarker.create_from_options(_face_landmarker_options)

# Hybrid: OpenCV HOG People Detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Camera Global
cap = None
is_fullscreen = False
window_name = "AI Lie Detector - Live Feed"

def init_camera():
    """Initialize webcam with optimized settings. Handles failures gracefully for cloud/headless environments."""
    global cap
    if cap is None:
        try:
            if os.name == 'nt':
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(0)

            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if 'DISPLAY' in os.environ or os.name == 'nt':
                    try:
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, 800, 600)
                    except: pass
                
                time.sleep(1.0)
                print("🎥 Webcam Initialized Successfully")
            else:
                print("⚠️ Warning: Physical Webcam not found. Using client-side capture only.")
                cap = None
        except Exception as e:
            print(f"⚠️ Warning: Camera initialization failed ({e}). Using client-side capture only.")
            cap = None

def calculate_ear(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio (EAR)."""
    try:
        p1 = np.array(landmarks[eye_indices[1]])
        p6 = np.array(landmarks[eye_indices[5]])
        p2 = np.array(landmarks[eye_indices[2]])
        p5 = np.array(landmarks[eye_indices[4]])
        v1 = np.linalg.norm(p1 - p6)
        v2 = np.linalg.norm(p2 - p5)
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
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
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
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(np.hstack((rmat, translation_vector)))
    yaw = euler_angles[1][0]
    return abs(yaw)

def process_image(frame):
    """Headless processing of a single frame (no camera required)."""
    if frame is None:
        return np.zeros(10, float), None

    # 1. FACE DETECTION using new Tasks API
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detect_results = face_detector.detect(mp_image)
    
    num_faces = 0
    if detect_results.detections:
        num_faces = len(detect_results.detections)
        for detection in detect_results.detections:
            bbox = detection.bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # 2. BODY DETECTION (HOG) - skip on low-power environments if many faces already found
    num_bodies = 0
    if num_faces < 1: # Only run HOG if no faces found to save CPU
        bodies, weights = hog.detectMultiScale(frame, winStride=(16, 16), padding=(8, 8), scale=1.1)
        for i, (x, y, w, h) in enumerate(bodies):
            if len(weights) > i and weights[i] > 0.6 and w > 60 and h > 120:
                num_bodies += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    final_count = max(num_faces, num_bodies)

    features = np.zeros(10, float)
    features[5] = float(final_count)
    return features, frame

def get_video_features_fast():
    """Optimized streaming version - face detection + HOG body detection, no landmarks."""
    init_camera()
    if cap is None:
        return np.zeros(10, float), None
    ret, frame = cap.read()
    if not ret:
        return np.zeros(10, float), None

    # 1. FACE DETECTION
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detect_results = face_detector.detect(mp_image)

    num_faces = 0
    if detect_results.detections:
        num_faces = len(detect_results.detections)
        for detection in detect_results.detections:
            bbox = detection.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # 2. BODY DETECTION (HOG) for people facing away or at distance
    bodies, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
    num_bodies = 0
    for i, (x, y, w, h) in enumerate(bodies):
        if len(weights) > i and weights[i] > 0.5 and w > 50 and h > 100:
            num_bodies += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)

    final_count = max(num_faces, num_bodies)

    features = np.zeros(10, float)
    features[5] = float(final_count)
    return features, frame

def get_video_features():
    init_camera()
    if cap is None:
        return np.zeros(10, float), None
    ret, frame = cap.read()
    if not ret:
        return np.zeros(10, float), None

    # 1. FACE DETECTION using new Tasks API
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detect_results = face_detector.detect(mp_image)
    
    num_faces = 0
    if detect_results.detections:
        num_faces = len(detect_results.detections)
        for detection in detect_results.detections:
            bbox = detection.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # 2. BODY DETECTION (HOG) 
    bodies, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
    num_bodies = 0
    for i, (x, y, w, h) in enumerate(bodies):
        if len(weights) > i and weights[i] > 0.5 and w > 50 and h > 100:
            num_bodies += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    final_count = max(num_faces, num_bodies)

    features = np.zeros(10, float)
    features[5] = float(final_count)
    return features, frame

def draw_overlay(frame, label=None, prob=None, drowsy=None, speaking_flag=None, look_away_flag=None, intruder_status=False, num_faces=0):
    """No-op: overlay text removed for clean video feed."""
    pass

def release_video():
    global cap
    if cap is not None:
        cap.release()
        cv2.destroyAllWindows()
        print("🎥 Camera Released")
