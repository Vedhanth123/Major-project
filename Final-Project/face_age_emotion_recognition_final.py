import cv2
import face_recognition
import os
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
import numpy as np
from deepface import DeepFace
import time
from fer import FER
from collections import Counter

# Create a folder named 'known_faces' and put your reference images there

# Load known faces and their names and estimated age ranges
known_face_encodings = []
known_face_names = []
known_faces_age_ranges = {}
known_faces_dir = 'known_faces'

def get_age_range(age):
    if age is None:
        return "Unknown Age"
    elif age < 10:
        return "<10"
    elif 10 <= age < 20:
        return "10-19"
    elif 20 <= age < 30:
        return "20-29"
    elif 30 <= age < 40:
        return "30-39"
    elif 40 <= age < 50:
        return "40-49"
    elif 50 <= age < 60:
        return "50-59"
    else:
        return "60+"

for filename in os.listdir(known_faces_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        if len(face_encodings) > 0:
            known_face_encodings.append(face_encodings[0])
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)
            try:
                age_predictions = DeepFace.analyze(img_path=image_path, actions=['age'], silent=True)
                if age_predictions and len(age_predictions) > 0:
                    estimated_age = age_predictions[0]['age']
                    known_faces_age_ranges[name] = get_age_range(estimated_age)
                else:
                    known_faces_age_ranges[name] = "Unknown Age"
                    print(f"Warning: Could not estimate age for {filename}")
            except Exception as e:
                known_faces_age_ranges[name] = "Unknown Age"
                print(f"Warning: Age estimation failed for {filename}: {e}")
        else:
            print(f"Warning: No face found in {filename}")

# Initialize webcam
video_capture = cv2.VideoCapture(0)
emotion_detector = FER()

def add_new_face(frame, face_location):
    """Opens a dialog to get the name of an unknown face and save it."""
    top, right, bottom, left = face_location
    face_image = frame[top:bottom, left:right]
    face_image_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

    root = tk.Tk()
    root.withdraw()

    name = simpledialog.askstring("Identify Face", "Enter the name of this person:", parent=root)

    if name:
        name = name.strip()
        if name:
            save_path = os.path.join(known_faces_dir, f"{name}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(np.array(face_image_pil), cv2.COLOR_RGB2BGR))
            print(f"Saved new face as {name}.jpg")

            new_image = face_recognition.load_image_file(save_path)
            new_encoding = face_recognition.face_encodings(new_image)
            if len(new_encoding) > 0:
                known_face_encodings.append(new_encoding[0])
                known_face_names.append(name)
                try:
                    age_predictions = DeepFace.analyze(img_path=save_path, actions=['age'], silent=True)
                    if age_predictions and len(age_predictions) > 0:
                        estimated_age = age_predictions[0]['age']
                        known_faces_age_ranges[name] = get_age_range(estimated_age)
                    else:
                        known_faces_age_ranges[name] = "Unknown Age"
                        print(f"Warning: Could not estimate age for newly saved face {name}")
                except Exception as e:
                    known_faces_age_ranges[name] = "Unknown Age"
                    print(f"Warning: Age estimation failed for newly saved face {name}: {e}")
            else:
                print(f"Error: No face found in the saved image {name}.jpg")

    root.destroy()

start_time = time.time()
run_duration = 8
emotion_history = {} # Dictionary to store emotion history for each detected face

while True:
    current_time = time.time()
    elapsed_time = current_time - start_time

    ret, frame = video_capture.read()
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_encodings_this_frame = list(face_encodings) # Create a copy for iteration

    for face_encoding in face_encodings_this_frame:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        recognized_age_range = None
        detected_emotion = None
        face_index = face_encodings_this_frame.index(face_encoding)
        top, right, bottom, left = face_locations[face_index]

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            recognized_age_range = known_faces_age_ranges.get(name)

            # Detect emotion for the recognized face
            face_image = frame[top:bottom, left:right]
            emotion_result = emotion_detector.detect_emotions(face_image)
            if emotion_result:
                emotions = emotion_result[0]['emotions']
                detected_emotion = max(emotions, key=emotions.get)

                if name not in emotion_history:
                    emotion_history[name] = []
                emotion_history[name].append(detected_emotion)

        else:
            add_new_face(frame, (top, right, bottom, left))
            name = "Recognizing..."

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Display current emotion above the bounding box
        if detected_emotion:
            cv2.putText(frame, f"Emotion: {detected_emotion}", (left + 6, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)

        # Draw the name and age below the bounding box
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, f"{name}, {recognized_age_range if recognized_age_range else 'Unknown Age'}", (left + 6, bottom - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or elapsed_time >= run_duration:
        break

video_capture.release()
cv2.destroyAllWindows()

print(f"\n--- Final Analysis (over {run_duration} seconds) ---")
for name, emotions in emotion_history.items():
    dominant_emotion = "Not Detected"
    if emotions:
        emotion_counts = Counter(emotions)
        dominant_emotion = emotion_counts.most_common(1)[0][0]

    age_range = known_faces_age_ranges.get(name, "Unknown Age")

    print(f"- Name: {name}, Age: {age_range}, Dominant Emotion: {dominant_emotion}")