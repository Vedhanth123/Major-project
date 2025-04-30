import tkinter as tk
from tkinter import messagebox
import face_recognition
import cv2
import os
import datetime
import threading

SAVE_DIR = 'known_faces'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Global variables
cap = None
frame = None
running = False
camera_thread = None

def start_camera():
    global cap, running, camera_thread
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
    
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to access the camera!")
        return
    
    running = True
    camera_thread = threading.Thread(target=show_camera, daemon=True)
    camera_thread.start()

def show_camera():
    global cap, frame, running
    while running:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_camera()
            break

def capture_face(name):
    global frame
    if frame is None:
        messagebox.showerror("Error", "Camera not started!")
        return

    if not name:
        messagebox.showwarning("Warning", "Please enter your name!")
        return

    face_locations = face_recognition.face_locations(frame)
    if len(face_locations) == 0:
        messagebox.showwarning("Warning", "No face detected! Please adjust.")
        return

    top, right, bottom, left = face_locations[0]
    face_image = frame[top:bottom, left:right]

    # Generate a timestamp and name-based filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(SAVE_DIR, f"{name}_{timestamp}.jpg")

    # Save the face image
    cv2.imwrite(save_path, face_image)
    messagebox.showinfo("Success", f"Face saved as {save_path}")

def stop_camera():
    global cap, running
    running = False
    if cap:
        cap.release()
    cv2.destroyAllWindows()

def on_closing():
    global camera_thread
    stop_camera()
    if camera_thread and camera_thread.is_alive():
        camera_thread.join()  # Ensure the camera thread finishes before quitting
    root.quit()

# Tkinter GUI
root = tk.Tk()
root.title("Face Capture App")

# Name label and entry
name_label = tk.Label(root, text="Enter your name:")
name_label.pack(pady=10)

name_entry = tk.Entry(root, width=30)
name_entry.pack(pady=10)

# Buttons
start_btn = tk.Button(root, text="Start Camera", command=start_camera, width=20, height=2)
start_btn.pack(pady=10)

capture_btn = tk.Button(root, text="Capture Face", command=lambda: capture_face(name_entry.get()), width=20, height=2)
capture_btn.pack(pady=10)

quit_btn = tk.Button(root, text="Quit", command=on_closing, width=20, height=2)
quit_btn.pack(pady=10)

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
