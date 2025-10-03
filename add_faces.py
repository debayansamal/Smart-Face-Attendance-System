import cv2
import pickle
import numpy as np
import os
import time

DATA_PATH = 'data/'
HAARCASCADE_PATH = os.path.join(DATA_PATH, 'haarcascade_frontalface_default.xml')
NAMES_PKL = os.path.join(DATA_PATH, 'names.pkl')
FACES_DATA_PKL = os.path.join(DATA_PATH, 'faces_data.pkl')

def add_new_face(name):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    video = cv2.VideoCapture(0)
    time.sleep(1)
    if not video.isOpened():
        return f"ERROR: Could not open camera (VideoCapture(0))."
    facedetect = cv2.CascadeClassifier(HAARCASCADE_PATH)
    faces_data = []
    i = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(faces_data) < 100 and i % 10 == 0:
                faces_data.append(resized_img)
            i += 1
            cv2.putText(frame, f"Capturing: {name} ({len(faces_data)}/100)", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
        flipped_frame = cv2.flip(frame, 1)
        cv2.imshow(f"Adding Face for {name} - Press 'q' to stop early", flipped_frame)
        k = cv2.waitKey(1)
        if k == ord('q') or len(faces_data) == 100:
            break
    video.release()
    cv2.destroyAllWindows()
    if len(faces_data) < 100:
        return f"ERROR: Captured only {len(faces_data)} images. Enrollment failed (requires 100 samples)."
    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape(len(faces_data), -1)
    if not os.path.exists(NAMES_PKL):
        names_list = [name] * len(faces_data)
        with open(NAMES_PKL, 'wb') as f:
            pickle.dump(names_list, f)
    else:
        with open(NAMES_PKL, 'rb') as f:
            names_list = pickle.load(f)
        names_list = names_list + [name] * len(faces_data)
        with open(NAMES_PKL, 'wb') as f:
            pickle.dump(names_list, f)
    if not os.path.exists(FACES_DATA_PKL):
        with open(FACES_DATA_PKL, 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open(FACES_DATA_PKL, 'rb') as f:
            faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)
        with open(FACES_DATA_PKL, 'wb') as f:
            pickle.dump(faces, f)
    return f"Successfully added {len(faces_data)} face samples for {name}. Remember to restart the server!"
