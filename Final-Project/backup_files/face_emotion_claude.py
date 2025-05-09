import face_recognition
import cv2
import numpy as np
import threading
import time
from queue import Queue
import os
import pickle
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
from fer import FER

class FaceRecognitionSystem:
    def __init__(self, known_faces_dir="known_faces", encodings_file="face_encodings.pkl"):
        # Directory to store known face images
        self.known_faces_dir = known_faces_dir
        self.encodings_file = encodings_file
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
        
        # Load existing encodings if available
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        
        # Create queues for thread communication
        self.frame_queue = Queue(maxsize=5)
        self.results_queue = Queue(maxsize=5)
        
        # For new face registration
        self.pending_faces = []  # List of (encoding, location, frame) tuples for unknown faces
        self.currently_registering = False
        
        # Initialize emotion detector
        try:
            self.emotion_detector = FER(mtcnn=True)  # Use MTCNN for better face detection
            self.emotion_detection_enabled = True
            print("Emotion detection initialized successfully")
        except Exception as e:
            print(f"Error initializing emotion detector: {e}")
            self.emotion_detection_enabled = False
        
        # Threading control
        self.is_running = False
        self.processing_thread = None
        self.gui_thread = None
        
        # Tkinter components
        self.root = None
        self.panel = None
        self.current_frame = None
        
        # Emotion color map (for displaying emotions with different colors)
        self.emotion_colors = {
            'angry': (0, 0, 255),     # Red
            'disgust': (0, 140, 255),  # Orange
            'fear': (0, 0, 128),      # Dark red
            'happy': (0, 255, 0),     # Green
            'sad': (255, 0, 0),       # Blue
            'surprise': (255, 255, 0), # Cyan
            'neutral': (255, 255, 255) # White
        }
        
    def load_known_faces(self):
        """Load known face encodings from file"""
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print(f"Loaded {len(self.known_face_names)} known faces")
            except Exception as e:
                print(f"Error loading known faces: {e}")
        else:
            print("No existing face database found. Will create new.")
            
    def save_known_faces(self):
        """Save known face encodings to file"""
        with open(self.encodings_file, 'wb') as f:
            pickle.dump({
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }, f)
        print(f"Saved {len(self.known_face_names)} known faces")
        
    def register_new_face(self, name, encoding, face_img):
        """Register a new face with a name"""
        # Add to known faces
        self.known_face_encodings.append(encoding)
        self.known_face_names.append(name)
        
        # Save the face image
        if face_img is not None:
            # Sanitize filename
            safe_name = ''.join(c if c.isalnum() else '_' for c in name)
            count = len([f for f in os.listdir(self.known_faces_dir) if f.startswith(safe_name)])
            filename = f"{safe_name}_{count}.jpg"
            cv2.imwrite(os.path.join(self.known_faces_dir, filename), face_img)
        
        # Save the updated encodings
        self.save_known_faces()
        
    def start(self):
        """Start the face recognition system"""
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Create and start GUI
        self.root = tk.Tk()
        self.root.title("Face Recognition with Emotion Detection")
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        
        # Add components to the GUI
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video feed panel
        self.panel = tk.Label(main_frame)
        self.panel.pack(padx=10, pady=10)
        
        # Status area
        status_frame = tk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = tk.Label(status_frame, text="System started. Waiting for faces...", 
                                     font=("Arial", 10), fg="blue")
        self.status_label.pack(side=tk.LEFT)
        
        # Emotion detection toggle
        if hasattr(self, 'emotion_detector'):
            emotion_var = tk.BooleanVar(value=self.emotion_detection_enabled)
            emotion_check = tk.Checkbutton(status_frame, text="Enable Emotion Detection", 
                                          variable=emotion_var, 
                                          command=lambda: setattr(self, 'emotion_detection_enabled', emotion_var.get()))
            emotion_check.pack(side=tk.RIGHT)
        
        # Control buttons
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        quit_btn = tk.Button(btn_frame, text="Quit", command=self.stop, 
                           bg="red", fg="white", font=("Arial", 10))
        quit_btn.pack(side=tk.RIGHT, padx=5)
        
        # Add legend for emotions
        if self.emotion_detection_enabled:
            legend_frame = tk.Frame(main_frame)
            legend_frame.pack(fill=tk.X, pady=5)
            
            legend_label = tk.Label(legend_frame, text="Emotion Colors:", font=("Arial", 10, "bold"))
            legend_label.pack(side=tk.LEFT, padx=5)
            
            for emotion, color in self.emotion_colors.items():
                # Convert BGR to RGB for Tkinter
                rgb_color = f'#{color[2]:02x}{color[1]:02x}{color[0]:02x}'
                emotion_label = tk.Label(legend_frame, text=emotion, bg=rgb_color, 
                                        font=("Arial", 8), padx=5, relief=tk.RAISED)
                emotion_label.pack(side=tk.LEFT, padx=2)
        
        # Start webcam capture in separate thread
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Start GUI update loop
        self._update_gui()
        self.root.mainloop()
        
    def stop(self):
        """Stop the face recognition system"""
        self.is_running = False
        if self.root:
            self.root.quit()
            self.root.destroy()
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
            
    def _capture_frames(self):
        """Capture frames from webcam in separate thread"""
        # Initialize webcam
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            self.stop()
            return
        
        try:
            while self.is_running:
                # Capture frame from webcam
                ret, frame = video_capture.read()
                if not ret:
                    break
                    
                # Put frame in queue for processing
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                
                time.sleep(0.01)  # Small delay to prevent overwhelming the CPU
        finally:
            # Release resources
            video_capture.release()
            
    def _process_frames(self):
        """Process frames for face recognition in a separate thread"""
        while self.is_running:
            if not self.frame_queue.empty():
                # Get a frame from the queue
                frame = self.frame_queue.get()
                
                # Analyze emotions if enabled
                emotions = {}
                if self.emotion_detection_enabled:
                    try:
                        # Detect emotions
                        emotion_results = self.emotion_detector.detect_emotions(frame)
                        if emotion_results:
                            # Map emotion results to face locations for later use
                            for result in emotion_results:
                                box = result["box"]
                                x, y, w, h = box
                                # Convert to face_recognition format (top, right, bottom, left)
                                loc = (y, x+w, y+h, x)
                                emotions[loc] = result["emotions"]
                    except Exception as e:
                        print(f"Error in emotion detection: {e}")
                
                # Resize frame for faster face recognition
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB
                
                # Find all faces and face encodings
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                new_face_indices = []
                
                for i, face_encoding in enumerate(face_encodings):
                    # Compare face with known faces
                    name = "Unknown"
                    
                    if len(self.known_face_encodings) > 0:
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                        
                        # Use the known face with the smallest distance to the new face
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index] and face_distances[best_match_index] < 0.6:  # Threshold for confidence
                            name = self.known_face_names[best_match_index]
                        else:
                            # This is a new face
                            new_face_indices.append(i)
                    else:
                        # First face in the system
                        new_face_indices.append(i)
                    
                    face_names.append(name)
                
                # Scale back face locations
                face_locations = [(top*4, right*4, bottom*4, left*4) 
                                  for top, right, bottom, left in face_locations]
                
                # Process emotions with face recognition results
                face_emotions = []
                for loc in face_locations:
                    # Try to find closest matching emotion detection
                    best_match = None
                    min_distance = float('inf')
                    
                    for emotion_loc, emotion_data in emotions.items():
                        # Calculate distance between face locations
                        distance = sum((a - b)**2 for a, b in zip(loc, emotion_loc))
                        if distance < min_distance:
                            min_distance = distance
                            best_match = emotion_data
                    
                    if best_match:
                        # Get the dominant emotion
                        dominant_emotion = max(best_match.items(), key=lambda x: x[1])[0]
                        emotion_value = best_match[dominant_emotion]
                        face_emotions.append((dominant_emotion, emotion_value))
                    else:
                        face_emotions.append((None, 0))
                
                # Check if we have new faces to register
                if not self.currently_registering:
                    for idx in new_face_indices:
                        # Get face image
                        top, right, bottom, left = face_locations[idx]
                        face_img = frame[top:bottom, left:right].copy()
                        
                        # Add to pending faces queue if not already there
                        face_encoding = face_encodings[idx]
                        if not any(np.array_equal(face_encoding, enc) for enc, _, _ in self.pending_faces):
                            self.pending_faces.append((face_encoding, face_locations[idx], face_img))
                
                # Put results in queue
                self.results_queue.put((face_locations, face_names, face_emotions, frame.copy()))
            else:
                # Sleep briefly to avoid consuming too much CPU
                time.sleep(0.01)
                
    def _update_gui(self):
        """Update the GUI with the latest frame and handle registration"""
        if not self.is_running:
            return
            
        # Process pending face registrations
        if self.pending_faces and not self.currently_registering:
            self.currently_registering = True
            self._register_next_face()
        
        # Update frame if available
        if not self.results_queue.empty():
            face_locations, face_names, face_emotions, frame = self.results_queue.get()
            
            # Display results
            for i, ((top, right, bottom, left), name, (emotion, confidence)) in enumerate(zip(face_locations, face_names, face_emotions)):
                # Select color based on known/unknown status
                if name != "Unknown":
                    box_color = (0, 255, 0)  # Green for known faces
                else:
                    box_color = (0, 0, 255)  # Red for unknown faces
                
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                
                # Prepare label text
                label_text = name
                if emotion:
                    # Get color for emotion
                    emotion_color = self.emotion_colors.get(emotion, (255, 255, 255))
                    label_text += f" ({emotion})"
                
                # Draw a label with the name below the face
                cv2.rectangle(frame, (left, bottom), (right, bottom + 35), box_color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, label_text, (left + 6, bottom + 25), font, 0.8, (255, 255, 255), 1)
                
                # Draw emotion indicator if detected
                if emotion and confidence > 0.4:  # Only show confident predictions
                    emotion_color = self.emotion_colors.get(emotion, (255, 255, 255))
                    cv2.rectangle(frame, (left, top - 20), (right, top), emotion_color, cv2.FILLED)
                    cv2.putText(frame, f"{emotion}: {confidence:.2f}", (left + 6, top - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Convert to format suitable for tkinter
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(self.current_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update the panel
            self.panel.configure(image=imgtk)
            self.panel.image = imgtk  # Keep a reference to prevent garbage collection
            
        # Schedule the next update
        if self.is_running and self.root:
            self.root.after(10, self._update_gui)
    
    def _register_next_face(self):
        """Register the next pending face with a dialog"""
        if not self.pending_faces:
            self.currently_registering = False
            return
            
        # Get the next face to register
        face_encoding, face_location, face_img = self.pending_faces.pop(0)
        
        # Convert face image for display
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_img_rgb)
        
        # Create a new dialog window for registration
        dialog = tk.Toplevel(self.root)
        dialog.title("Register New Face")
        dialog.geometry("300x400")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()  # Make dialog modal
        
        # Display the face image
        face_width, face_height = face_pil.size
        max_size = 200
        scale = min(max_size/face_width, max_size/face_height)
        new_size = (int(face_width*scale), int(face_height*scale))
        face_pil = face_pil.resize(new_size, Image.Resampling.LANCZOS)
        
        face_tk = ImageTk.PhotoImage(face_pil)
        img_label = tk.Label(dialog, image=face_tk)
        img_label.image = face_tk  # Keep a reference
        img_label.pack(pady=10)
        
        # Add prompt and entry
        tk.Label(dialog, text="Enter name for this person:", font=("Arial", 12)).pack(pady=5)
        name_var = tk.StringVar()
        name_entry = tk.Entry(dialog, textvariable=name_var, font=("Arial", 12), width=20)
        name_entry.pack(pady=5)
        name_entry.focus_set()
        
        # Function to handle registration
        def on_register():
            name = name_var.get().strip()
            if name:
                self.register_new_face(name, face_encoding, face_img)
                self.status_label.config(text=f"Registered new face: {name}")
                dialog.destroy()
                # Process next face after a short delay
                self.root.after(500, self._register_next_face)
            else:
                messagebox.showwarning("Warning", "Please enter a name", parent=dialog)
        
        # Function to skip this face
        def on_skip():
            dialog.destroy()
            # Process next face after a short delay
            self.root.after(500, self._register_next_face)
        
        # Add buttons
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(btn_frame, text="Register", command=on_register, 
                 bg="green", fg="white", font=("Arial", 12)).pack(side=tk.LEFT, expand=True, padx=5)
        tk.Button(btn_frame, text="Skip", command=on_skip,
                 bg="gray", fg="white", font=("Arial", 12)).pack(side=tk.RIGHT, expand=True, padx=5)
        
        # Bind Enter key to register
        dialog.bind("<Return>", lambda event: on_register())
        
        # Center the dialog on the parent window
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (width // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (height // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")

# Example usage
if __name__ == "__main__":
    face_system = FaceRecognitionSystem()
    face_system.start()