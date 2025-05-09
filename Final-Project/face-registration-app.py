import streamlit as st
import cv2
import face_recognition
import os
import time
import sqlite3
from PIL import Image
import numpy as np

# Configuration
KNOWN_FACES_DIR = 'known_faces'
DATABASE_PATH = "user_data.db"
FACE_DETECTION_DURATION = 5  # Default seconds to run face detection

# Create directory for known faces if it doesn't exist
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)
    st.write(f"Created '{KNOWN_FACES_DIR}' directory")

def load_known_faces():
    """Load known faces from the directory and return their names"""
    known_face_names = []
    
    if not os.path.exists(KNOWN_FACES_DIR):
        return []
        
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)
                
    return known_face_names

def create_database():
    """Create the database and necessary tables if they don't exist"""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Create user preferences table (minimal version)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            name TEXT PRIMARY KEY,
            favorite_genre TEXT,
            preferred_duration TEXT,
            last_updated INTEGER
        )
        ''')
        
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Database creation error: {e}")
        return False
    finally:
        if conn:
            conn.close()

def add_user_to_database(name):
    """Add a new user to the database with default preferences"""
    conn = None
    try:
        # Ensure database exists
        create_database()
            
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Check if user already exists
        cursor.execute('SELECT name FROM user_preferences WHERE name = ?', (name,))
        if cursor.fetchone() is None:
            # Insert new user with default preferences
            cursor.execute('''
            INSERT INTO user_preferences (name, favorite_genre, preferred_duration, last_updated)
            VALUES (?, ?, ?, ?)
            ''', (name, "general", "medium", int(time.time())))
            
            conn.commit()
            return True
        return True  # User already exists
    except Exception as e:
        st.error(f"Database error: {e}")
        return False
    finally:
        if conn:
            conn.close()

def capture_and_save_face(frame, name):
    """Process frame to detect, crop, and save a face"""
    # Convert the image to RGB (face_recognition uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Find all the faces in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    
    if not face_locations:
        return False, "No face detected in the frame"
    
    if len(face_locations) > 1:
        return False, "Multiple faces detected. Please ensure only one face is in the frame."
    
    # Get the location of the face
    top, right, bottom, left = face_locations[0]
    
    # Crop the face from the frame with a small margin
    margin = 30
    height, width = frame.shape[:2]
    
    # Ensure margins don't go outside image boundaries
    top_margin = max(0, top - margin)
    bottom_margin = min(height, bottom + margin)
    left_margin = max(0, left - margin)
    right_margin = min(width, right + margin)
    
    face_image = frame[top_margin:bottom_margin, left_margin:right_margin]
    
    # Save the face image
    save_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    cv2.imwrite(save_path, face_image)
    
    # Verify that the image contains a detectable face
    try:
        test_image = face_recognition.load_image_file(save_path)
        face_encodings = face_recognition.face_encodings(test_image)
        if not face_encodings:
            os.remove(save_path)  # Remove the file if face detection failed
            return False, "Failed to encode face. Please try again with better lighting."
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        return False, f"Error processing face: {str(e)}"
    
    # Add user to database
    add_user_to_database(name)
    
    return True, save_path

def capture_from_webcam(stframe, status_text, progress_bar, name, duration=FACE_DETECTION_DURATION):
    """Capture video from webcam for a set duration to get a good face image"""
    # Initialize webcam
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        status_text.error("Error: Could not open webcam")
        return False, None
        
    best_frame = None
    start_time = time.time()
    frames_captured = []
    
    status_text.info(f"Looking for the best face shot for {duration} seconds...")
    
    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time
        remaining_time = max(0, duration - elapsed_time)
        
        # Update progress bar
        progress = min(1.0, elapsed_time / duration)
        progress_bar.progress(progress)
        
        # Break if time exceeded
        if elapsed_time >= duration:
            break

        ret, frame = video_capture.read()
        if not ret:
            status_text.error("Error: Could not read frame from webcam")
            continue
        
        # Convert to RGB for display and processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if face_locations:
            frames_captured.append(frame.copy())
            
            # Draw rectangle around face
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Add countdown text
        cv2.putText(frame, f"Time: {int(remaining_time)}s", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the frame in Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    
    video_capture.release()
    
    # Process frames to find the best one
    if not frames_captured:
        return False, "No faces detected during the capture period"
    
    # For simplicity, we'll just use the last frame where a face was detected
    # In a more sophisticated app, you could implement quality metrics
    best_frame = frames_captured[-1]
    
    # Save the face
    success, result = capture_and_save_face(best_frame, name)
    return success, result

def upload_and_process_image(uploaded_file, name):
    """Process an uploaded image file to extract and save a face"""
    try:
        # Read the image file
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        # Convert to BGR (OpenCV format) if it's RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Save the face
        success, result = capture_and_save_face(image, name)
        return success, result
    
    except Exception as e:
        return False, f"Error processing image: {str(e)}"

def delete_face(name):
    """Delete a face from the system"""
    try:
        # Delete the image file
        file_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Error deleting face: {e}")
        return False

def main():
    st.set_page_config(
        page_title="Face Registration System",
        page_icon="👤",
        layout="wide",
    )
    
    # App title and description
    st.title("Face Registration System")
    st.markdown("Add, view, and manage facial identities for the emotion-based recommendation system.")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Add Face", "Manage Faces", "Help"])
    
    # Tab 1: Add Face
    with tab1:
        st.header("Add New Face")
        
        # User input
        name = st.text_input("Enter name for the new face")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Option 1: Capture from Webcam")
            webcam_duration = st.slider("Capture Duration (seconds)", 
                               min_value=3, max_value=15, value=5)
            
            capture_btn = st.button("Capture from Webcam", use_container_width=True)
            
            if capture_btn:
                if not name or name.strip() == "":
                    st.error("Please enter a name first")
                else:
                    # Check if name already exists
                    existing_names = load_known_faces()
                    if name in existing_names:
                        st.warning(f"A face with the name '{name}' already exists. Please use a different name or delete the existing face first.")
                    else:
                        # Set up capture UI
                        video_placeholder = st.empty()
                        status_text = st.empty()
                        progress_bar = st.progress(0)
                        
                        # Capture face
                        success, result = capture_from_webcam(video_placeholder, status_text, progress_bar, name, webcam_duration)
                        
                        if success:
                            st.success(f"Face for {name} registered successfully!")
                            # Display the captured and saved face
                            image = cv2.imread(result)
                            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"{name}'s registered face")
                        else:
                            st.error(f"Failed to register face: {result}")
        
        with col2:
            st.subheader("Option 2: Upload Image")
            uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
            
            upload_btn = st.button("Process Uploaded Image", use_container_width=True)
            
            if upload_btn and uploaded_file is not None:
                if not name or name.strip() == "":
                    st.error("Please enter a name first")
                else:
                    # Check if name already exists
                    existing_names = load_known_faces()
                    if name in existing_names:
                        st.warning(f"A face with the name '{name}' already exists. Please use a different name or delete the existing face first.")
                    else:
                        # Process the uploaded image
                        with st.spinner("Processing image..."):
                            success, result = upload_and_process_image(uploaded_file, name)
                            
                            if success:
                                st.success(f"Face for {name} registered successfully!")
                                # Display the captured and saved face
                                image = cv2.imread(result)
                                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"{name}'s registered face")
                            else:
                                st.error(f"Failed to register face: {result}")
    
    # Tab 2: Manage Faces
    with tab2:
        st.header("Manage Registered Faces")
        
        # Refresh button
        if st.button("Refresh Face List", key="refresh_faces"):
            st.experimental_rerun()
        
        # Load and display known faces
        known_faces = load_known_faces()
        
        if not known_faces:
            st.info("No faces registered yet. Go to the 'Add Face' tab to register faces.")
        else:
            st.success(f"{len(known_faces)} faces registered")
            
            # Display faces in a grid
            cols_per_row = 3
            rows = (len(known_faces) + cols_per_row - 1) // cols_per_row  # Ceiling division
            
            for row in range(rows):
                cols = st.columns(cols_per_row)
                for col in range(cols_per_row):
                    idx = row * cols_per_row + col
                    if idx < len(known_faces):
                        name = known_faces[idx]
                        with cols[col]:
                            image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
                            if os.path.exists(image_path):
                                image = cv2.imread(image_path)
                                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=name, width=200)
                                
                                # Delete button for each face
                                if st.button(f"Delete {name}", key=f"delete_{name}"):
                                    if delete_face(name):
                                        st.success(f"Deleted {name}'s face")
                                        time.sleep(1)  # Short delay to show the message
                                        st.experimental_rerun()
                                    else:
                                        st.error(f"Failed to delete {name}'s face")
                            else:
                                st.error(f"Image file for {name} not found")
    
    # Tab 3: Help
    with tab3:
        st.header("Help & Information")
        
        st.subheader("About this application")
        st.write("""
        This face registration system allows you to add and manage facial identities for the emotion-based video recommendation system.
        
        **Features:**
        - Register faces using webcam capture or image upload
        - View all registered faces
        - Delete faces from the system
        
        **Tips for good face registration:**
        - Ensure good lighting
        - Face the camera directly
        - Remove glasses if possible
        - Keep a neutral expression
        - Avoid extreme angles
        """)
        
        st.subheader("Troubleshooting")
        st.write("""
        **If face detection fails:**
        - Try better lighting
        - Move closer to the camera
        - Ensure your face is clearly visible
        - Try uploading a clear image instead
        
        **If webcam doesn't work:**
        - Allow browser permission for camera access
        - Try restarting the application
        - Check if another application is using the camera
        """)

if __name__ == "__main__":
    main()
