# Smart Face Attendance System

A modern, deep learning-based attendance tracking system utilizing facial recognition for automated student/employee check-in. The system consists of a Python/Flask backend handling the computer vision (OpenCV/Face Recognition library) and a web frontend built with HTML, Tailwind CSS, and JavaScript.

---

## 🚀 Features

* **Real-time Face Scanning:** Live video feed for instant recognition.
* **User Enrollment:** Simple interface to add new users by capturing a face sample and associating it with a name.
* **Automatic Attendance Marking:** Single button click to record attendance for the currently recognized person.
* **Attendance Logging:** Displays today's attendance record with names and timestamps.
* **Data Management:** Tools to view, delete individual, or clear all enrolled face data.

---

## 🛠️ Tech Stack

### Frontend
* **HTML5 / Tailwind CSS:** Responsive and modern UI.
* **JavaScript (Fetch API):** Client-side logic for interacting with the backend API.

### Backend
* **Python (Flask):** Web framework to serve the frontend and handle API requests.
* **OpenCV:** Handles camera stream and image processing.
* **Face Recognition (Dlib):** Core library for facial encoding and matching.
