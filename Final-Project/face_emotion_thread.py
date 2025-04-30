import cv2
import face_recognition
import numpy as np
import os
import tkinter as tk
from tkinter import simpledialog
import uuid
import threading
from fer import FER

# Load known faces from directory
def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir('known_faces'):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            path = os.path.join('known_faces', filename)
            image = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_face_encodings.append(encoding[0])
                name = os.path.splitext(filename)[0]
                known_face_names.append(name)

    return known_face_encodings, known_face_names

# Ask for name and save unknown face
def ask_and_save_unknown_face(face_image):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    name = simpledialog.askstring("Unknown Face", "Enter name of the person:")
    root.destroy()

    if name:
        unique_name = f"{name}_{uuid.uuid4().hex[:6]}"
        save_path = os.path.join('known_faces', f"{unique_name}.jpg")
        cv2.imwrite(save_path, face_image)
        print(f"[INFO] Saved new face: {save_path}")
        return True
    return False

def face_recognition_and_emotion_thread():
    known_face_encodings, known_face_names = load_known_faces()
    emotion_detector = FER(mtcnn=True)

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            emotion = ""

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            else:
                top, right, bottom, left = [v * 4 for v in face_location]
                face_img = frame[top:bottom, left:right]
                if ask_and_save_unknown_face(face_img):
                    known_face_encodings, known_face_names = load_known_faces()

            # Detect emotion using FER on the full-sized face
            top, right, bottom, left = [v * 4 for v in face_location]
            face_img = frame[top:bottom, left:right]
            try:
                result = emotion_detector.detect_emotions(face_img)
                if result:
                    top_emotion = max(result[0]['emotions'], key=result[0]['emotions'].get)
                    emotion = top_emotion
            except Exception as e:
                print(f"[ERROR] Emotion detection failed: {e}")

            label = f"{name} | {emotion}" if emotion else name

            # Draw rectangle and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

        cv2.imshow('Face & Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    face_thread = threading.Thread(target=face_recognition_and_emotion_thread)
    face_thread.start()
    face_thread.join()
