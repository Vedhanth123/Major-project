import cv2
import face_recognition
import os
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
import numpy as np

# Create a folder named 'known_faces' and put your reference images there

# Load known faces and their names
known_face_encodings = []
known_face_names = []
known_faces_dir = 'known_faces'

for filename in os.listdir(known_faces_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        if len(face_encodings) > 0:
            known_face_encodings.append(face_encodings[0])
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)
        else:
            print(f"Warning: No face found in {filename}")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

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
            else:
                print(f"Error: No face found in the saved image {name}.jpg")

    root.destroy()

while True:
    ret, frame = video_capture.read()
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        else:
            add_new_face(frame, (top, right, bottom, left))
            name = "Recognizing..."

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label below the box
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()