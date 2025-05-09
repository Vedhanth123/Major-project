import tkinter as tk
from tkinter import messagebox
import face_recognition
import cv2
import datetime
import threading

# Initialize webcam and model
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    messagebox.showerror("Error", "Failed to access the camera!")
    exit()

# Emotion detection example
def detect_emotion(face):
    # Placeholder for emotion detection logic (use a pre-trained model like FER or OpenCV)
    return "Neutral"  # Example placeholder

def show_camera():
    global cap
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip image to create mirror effect
        frame = cv2.flip(frame, 1)
        face_locations = face_recognition.face_locations(frame)
        
        # Draw bounding boxes around faces and detect emotions
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            face_image = frame[top:bottom, left:right]
            emotion = detect_emotion(face_image)  # Placeholder function
            cv2.putText(frame, emotion, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Camera Feed', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start camera in a separate thread
def start_camera():
    camera_thread = threading.Thread(target=show_camera, daemon=True)
    camera_thread.start()

# Basic GUI with Tkinter
root = tk.Tk()
root.title("Face Capture & Emotion Detection")

start_btn = tk.Button(root, text="Start Camera", command=start_camera)
start_btn.pack(pady=20)

root.mainloop()
