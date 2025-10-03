import json
import pandas as pd
import os
from flask import Flask, request, jsonify, Response, send_from_directory
from datetime import datetime 

from add_faces import add_new_face
from face_recognition_stream import (
    generate_frames, 
    get_recognized_name, 
    mark_attendance_for_last_recognized,
    set_streaming_state,
    get_enrolled_names,
    delete_all_data,
    delete_individual_face,
    clear_today_attendance
)

ATTENDANCE_FOLDER = 'Attendance'

app = Flask(__name__, static_folder='static') 

def get_attendance_data():
    date = datetime.now().strftime("%d-%m-%Y")
    csv_filepath = os.path.join(ATTENDANCE_FOLDER, f"Attendance_{date}.csv")
    
    if not os.path.exists(csv_filepath):
        return []

    try:
        df = pd.read_csv(csv_filepath)
        data = df.to_dict('records')
        return data
    except pd.errors.EmptyDataError:
        return []
    except Exception as e:
        print(f"Error reading attendance CSV: {e}")
        return []

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/add_face', methods=['POST'])
def add_face_api():
    data = request.get_json()
    name = data.get('name')

    if not name:
        return jsonify({"status": "error", "message": "Name is required for enrollment"}), 400

    print(f"Starting face enrollment for: {name}")
    
    result_message = add_new_face(name)
    
    if "ERROR" in result_message:
        return jsonify({"status": "error", "message": result_message}), 500
    else:
        return jsonify({"status": "success", "message": result_message}), 200

@app.route('/api/start_scan', methods=['POST'])
def start_scan_api():
    set_streaming_state(True)
    return jsonify({"status": "success", "message": "Scan started."})

@app.route('/api/stop_scan', methods=['POST'])
def stop_scan_api():
    set_streaming_state(False)
    return jsonify({"status": "success", "message": "Scan stopped."})

@app.route('/api/mark_attendance', methods=['POST'])
def mark_attendance_api():
    print("Attempting to mark attendance...")
    result = mark_attendance_for_last_recognized()
    if result["status"] == "error":
        return jsonify(result), 400
    else:
        return jsonify(result), 200

@app.route('/api/recognized_name')
def recognized_name_api():
    name = get_recognized_name()
    return jsonify({"status": "success", "name": name})

@app.route('/api/attendance_data')
def attendance_data_api():
    data = get_attendance_data()
    return jsonify({"status": "success", "data": data})

@app.route('/api/manage/enrolled_names', methods=['GET'])
def enrolled_names_api():
    names = get_enrolled_names()
    return jsonify({"status": "success", "names": names})

@app.route('/api/manage/delete_all_faces', methods=['POST'])
def delete_all_faces_api():
    result = delete_all_data()
    return jsonify(result)

@app.route('/api/manage/delete_face/<name>', methods=['POST'])
def delete_face_api(name):
    result = delete_individual_face(name)
    return jsonify(result)

@app.route('/api/manage/clear_attendance', methods=['POST'])
def clear_attendance_api():
    result = clear_today_attendance()
    return jsonify(result)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    if not os.path.exists(ATTENDANCE_FOLDER):
        os.makedirs(ATTENDANCE_FOLDER)
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
