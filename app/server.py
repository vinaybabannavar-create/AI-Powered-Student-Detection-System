# app/server.py

import sqlite3, os, subprocess, cv2, sys, base64
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

# Import modules from parent directory
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from modules.video_module import get_video_features, get_video_features_fast, draw_overlay, process_image
from modules.fusion_model import predict_result
from modules.drowsiness_module import update_sleep_status

# ---------------- DATABASE SETUP ---------------- #

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database', 'users.db')
os.makedirs(os.path.join(BASE_DIR, 'database'), exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ---------------- FLASK APP ---------------- #

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'classroom-counter-secure-key'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ---------------- ROUTES ---------------- #

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        if not username or not password:
            flash('Username and password required', 'danger')
            return redirect(url_for('signup'))
        conn = get_db_connection()
        try:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                         (username, generate_password_hash(password)))
            conn.commit()
            conn.close()
            flash('Account created successfully. Please login.', 'success')
            return redirect(url_for('login'))
        except:
            conn.close()
            flash('Username already exists.', 'danger')
            return redirect(url_for('signup'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['user'] = username
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.', 'danger')
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out successfully.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session.get('user'), show_video=False)

@app.route('/start-session')
def start_session():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session.get('user'), show_video=True)

# ---------------- VIDEO STREAMING ---------------- #

last_count = 0

def gen_frames():
    global last_count
    while True:
        # Use fast version - only face detection, no heavy processing
        video_feat, frame = get_video_features_fast()
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "NO CAMERA SIGNAL", (180, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            video_feat = np.zeros(10)
        
        if video_feat is not None:
            last_count = int(video_feat[5])

        # Encode and yield - clean frame, no text overlay
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    if 'user' not in session:
        return "Unauthorized", 401
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global last_count
    try:
        import time
        start_t = time.time()

        # Accept both FormData (binary blob, v6) and JSON (base64, legacy)
        if 'frame' in request.files:
            # v6: Binary blob upload (faster, no base64 overhead)
            file = request.files['frame']
            img_bytes = file.read()
        else:
            # Legacy: base64 JSON upload
            data = request.get_json()
            image_data = data['image'].split(",")[1]
            img_bytes = base64.b64decode(image_data)

        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        video_feat, _, boxes = process_image(frame)
        duration = (time.time() - start_t) * 1000
        last_count = int(video_feat[5])

        print(f"DEBUG: {duration:.0f}ms | Count: {last_count} | Faces: {len(boxes)}")

        return jsonify({
            'success': True,
            'count': last_count,
            'boxes': boxes
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/current_count')
def current_count():
    return jsonify({'count': last_count})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
