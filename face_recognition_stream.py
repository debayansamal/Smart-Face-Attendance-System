import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from threading import Lock
from sklearn.neighbors import KNeighborsClassifier 

DATA_PATH = 'data/'
HAARCASCADE_PATH = os.path.join(DATA_PATH, 'haarcascade_frontalface_default.xml')
NAMES_PKL = os.path.join(DATA_PATH, 'names.pkl')
FACES_DATA_PKL = os.path.join(DATA_PATH, 'faces_data.pkl')
ATTENDANCE_FOLDER = 'Attendance'
COL_NAMES = ['NAME', 'TIME']

KNN_MODEL = None
LABELS = []
LAST_RECOGNIZED_NAME = "Initializing..."
STATE_LOCK = Lock()
VIDEO_CAPTURE = None
IS_STREAMING = False

def initialize_model():
    global KNN_MODEL, LABELS, LAST_RECOGNIZED_NAME 
    with STATE_LOCK:
        if not os.path.exists(NAMES_PKL) or not os.path.exists(FACES_DATA_PKL):
            LAST_RECOGNIZED_NAME = "Training data missing"
            print("MODEL ERROR: Training data files (.pkl) not found. Run 'Add Face' first.")
            return False
        try:
            with open(NAMES_PKL, 'rb') as w:
                LABELS = pickle.load(w)
            with open(FACES_DATA_PKL, 'rb') as f:
                FACES = pickle.load(f)
            if len(LABELS) == 0:
                 LAST_RECOGNIZED_NAME = "No faces enrolled"
                 KNN_MODEL = None
                 return False
            KNN_MODEL = KNeighborsClassifier(n_neighbors=5)
            KNN_MODEL.fit(FACES, LABELS)
            LAST_RECOGNIZED_NAME = "Ready"
            print(f"Model initialized successfully. Trained on {len(LABELS)} samples.")
            return True
        except Exception as e:
            LAST_RECOGNIZED_NAME = f"Model Error: {e}"
            print(f"MODEL ERROR: Failed to load training data: {e}")
            return False

def get_video_capture():
    global VIDEO_CAPTURE
    with STATE_LOCK:
        if VIDEO_CAPTURE is None or not VIDEO_CAPTURE.isOpened():
            VIDEO_CAPTURE = cv2.VideoCapture(0)
            time.sleep(1)
            if not VIDEO_CAPTURE.isOpened():
                print("CAMERA ERROR: Could not open camera (VideoCapture(0)).")
                return None
        return VIDEO_CAPTURE

def set_streaming_state(state):
    global IS_STREAMING
    with STATE_LOCK:
        IS_STREAMING = state
    if not state:
        set_recognized_name("Scan stopped")

def set_recognized_name(name):
    global LAST_RECOGNIZED_NAME
    with STATE_LOCK:
        LAST_RECOGNIZED_NAME = name

def get_recognized_name():
    with STATE_LOCK:
        return LAST_RECOGNIZED_NAME

def generate_frames():
    global IS_STREAMING
    if KNN_MODEL is None and not initialize_model():
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, get_recognized_name(), (50, 240), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 
               cv2.imencode('.jpg', error_frame)[1].tobytes() + 
               b'\r\n')
        return
    cap = get_video_capture()
    facedetect = cv2.CascadeClassifier(HAARCASCADE_PATH)
    if cap is None or facedetect.empty():
        set_recognized_name("Camera/Classifier Error")
        return 
    while True:
        if not IS_STREAMING:
            time.sleep(0.5)
            continue
        try:
            ret, frame = cap.read()
            if not ret:
                set_recognized_name("Camera disconnected.")
                set_streaming_state(False)
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)
            current_frame_recognition = "No face detected"
            for (x, y, w, h) in faces:
                crop_img = frame[y:y+h, x:x+w, :]
                resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
                output = KNN_MODEL.predict(resized_img)
                recognized_name = str(output[0])
                current_frame_recognition = recognized_name
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
                cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
                cv2.putText(frame, recognized_name, (x, y - 15),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            set_recognized_name(current_frame_recognition)
            frame = cv2.flip(frame, 1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.01)
        except Exception as e:
            print(f"Streaming error: {e}")
            set_streaming_state(False)
            break

def mark_attendance_for_last_recognized():
    name = get_recognized_name()
    invalid_names = ["No face detected", "Initializing...", "Ready", "Unknown", "Model Error", "Training data missing", "Scan stopped", "No faces enrolled"]
    if name in invalid_names or "Error" in name:
        return {"status": "error", "message": f"Cannot mark attendance. Current status: {name}. Please start scan and face the camera."}
    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
    timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
    if not os.path.exists(ATTENDANCE_FOLDER):
        os.makedirs(ATTENDANCE_FOLDER)
    csv_filepath = os.path.join(ATTENDANCE_FOLDER, f"Attendance_{date}.csv")
    exist = os.path.isfile(csv_filepath)
    attendance_entry = [name, timestamp]
    try:
        with open(csv_filepath, 'a', newline='') as csvfile: 
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(COL_NAMES)
            writer.writerow(attendance_entry)
        set_streaming_state(False)
        return {"status": "success", "message": f"Attendance marked for: {name} at {timestamp}"}
    except Exception as e:
        return {"status": "error", "message": f"ERROR: Could not write attendance to CSV: {e}"}

def clear_today_attendance():
    date = datetime.now().strftime("%d-%m-%Y")
    csv_filepath = os.path.join(ATTENDANCE_FOLDER, f"Attendance_{date}.csv")
    try:
        if os.path.exists(csv_filepath):
            os.remove(csv_filepath)
            return {"status": "success", "message": f"Attendance for {date} cleared."}
        else:
            return {"status": "success", "message": "No attendance file found to clear."}
    except Exception as e:
        return {"status": "error", "message": f"ERROR: Could not clear attendance file: {e}"}

def get_enrolled_names():
    if not os.path.exists(NAMES_PKL):
        return []
    try:
        with open(NAMES_PKL, 'rb') as w:
            LABELS = pickle.load(w)
        return sorted(list(set(LABELS)))
    except:
        return []

def delete_all_data():
    global KNN_MODEL, LABELS
    try:
        files = [NAMES_PKL, FACES_DATA_PKL]
        for f in files:
            if os.path.exists(f):
                os.remove(f)
        KNN_MODEL = None
        LABELS = []
        set_recognized_name("Data cleared. Restart server.")
        return {"status": "success", "message": "All enrolled face data cleared. Restart server to finalize."}
    except Exception as e:
        return {"status": "error", "message": f"ERROR: Could not delete data: {e}"}

def delete_individual_face(name_to_delete):
    global KNN_MODEL, LABELS
    try:
        with open(NAMES_PKL, 'rb') as w:
            LABELS = pickle.load(w)
        with open(FACES_DATA_PKL, 'rb') as f:
            FACES = pickle.load(f)
        indices_to_keep = [i for i, name in enumerate(LABELS) if name != name_to_delete]
        if len(indices_to_keep) == len(LABELS):
            return {"status": "error", "message": f"Name '{name_to_delete}' not found in data."}
        new_LABELS = [LABELS[i] for i in indices_to_keep]
        if len(new_LABELS) > 0:
            new_FACES = FACES[indices_to_keep]
            with open(FACES_DATA_PKL, 'wb') as f:
                pickle.dump(new_FACES, f)
        else:
            if os.path.exists(FACES_DATA_PKL):
                 os.remove(FACES_DATA_PKL)
        with open(NAMES_PKL, 'wb') as w:
            pickle.dump(new_LABELS, w)
        KNN_MODEL = None
        LABELS = new_LABELS
        set_recognized_name(f"Deleted {name_to_delete}. Restart server.")
        return {"status": "success", "message": f"Successfully deleted all samples for {name_to_delete}. Restart server to finalize."}
    except FileNotFoundError:
        return {"status": "error", "message": "Training data files not found."}
    except Exception as e:
        return {"status": "error", "message": f"ERROR: Could not delete individual face: {e}"}

initialize_model()
