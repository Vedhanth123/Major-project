import cv2
import os
import face_recognition
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk

# Create folder if it doesn't exist
SAVE_DIR = "known_faces"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_face():
    name = name_entry.get().strip()

    if not name:
        messagebox.showwarning("Missing Name", "Please enter a name before capturing.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Cannot access webcam.")
        return

    messagebox.showinfo("Instructions", "Press 's' to save your face or 'q' to cancel.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)

        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        cv2.imshow("Capture Face - Press 's' to Save", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if face_locations:
                top, right, bottom, left = face_locations[0]
                face_img = frame[top:bottom, left:right]
                file_path = os.path.join(SAVE_DIR, f"{name}.jpg")
                cv2.imwrite(file_path, face_img)
                messagebox.showinfo("Saved", f"Face saved as {file_path}")
            else:
                messagebox.showwarning("No Face", "No face detected. Try again.")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI
root = Tk()
root.title("Face Saver")
root.geometry("300x150")

Label(root, text="Enter Your Name:").pack(pady=10)
name_entry = Entry(root)
name_entry.pack(pady=5)

Button(root, text="Start Camera & Capture Face", command=save_face).pack(pady=20)

root.mainloop()
