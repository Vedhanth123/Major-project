
import cv2
import face_recognition
import os
import tkinter as tk
from tkinter import simpledialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
from deepface import DeepFace
import time
from fer import FER
from collections import Counter
import threading
import webbrowser
import sqlite3
import requests
from io import BytesIO
from googleapiclient.discovery import build
import random
from datetime import datetime, timedelta

# # Import our database functions
from database import log_emotion_data # get_user_preferences
from geo import get_geolocation

# Configuration
YOUTUBE_API_KEY = "AIzaSyAVer6TDzEBXthg6l4LtxJSMW6ybHGqB8c"  # Replace with your actual API key
DATABASE_PATH = "user_data.db" 
MAX_RESULTS = 10
FACE_DETECTION_DURATION = 8  # Seconds to run face detection

# Ensure database exists
# create_database()

# Emotion to video mapping
EMOTION_VIDEO_MAPPING = {
    "happy": {
        "keywords": ["uplifting music", "comedy videos", "funny moments", "feel good content"],
        "categories": ["22", "23", "24"],  # Music, Comedy, Entertainment
    },
    "sad": {
        "keywords": ["relaxing music", "calming videos", "motivational speeches", "inspirational stories"],
        "categories": ["22", "25", "27"],  # Music, News & Politics, Education
    },
    "angry": {
        "keywords": ["calming music", "meditation videos", "relaxation techniques", "nature sounds"],
        "categories": ["22", "28"],  # Music, Science & Technology
    },
    "fear": {
        "keywords": ["soothing music", "positive affirmations", "calming content", "guided relaxation"],
        "categories": ["22", "26"],  # Music, Howto & Style
    },
    "surprise": {
        "keywords": ["amazing facts", "incredible discoveries", "wow moments", "mind blowing"],
        "categories": ["28", "24", "27"],  # Science & Tech, Entertainment, Education
    },
    "neutral": {
        "keywords": ["interesting documentaries", "educational content", "informative videos", "how-to guides"],
        "categories": ["27", "28", "26"],  # Education, Science & Tech, Howto & Style
    },
    "disgust": {
        "keywords": ["satisfying videos", "clean organization", "aesthetic content", "art videos"],
        "categories": ["24", "26"],  # Entertainment, Howto & Style
    }
}

# Age range to content type mapping
AGE_CONTENT_MAPPING = {
    "<10": {
        "append_keywords": ["for kids", "child friendly", "educational"],
        "restrict": "strict",  # Apply strict content filtering
    },
    "10-19": {
        "append_keywords": ["teen", "trending"],
        "restrict": "moderate",  # Apply moderate content filtering
    },
    "20-29": {
        "append_keywords": ["trending", "popular"],
        "restrict": "none",  # No special content filtering
    },
    "30-39": {
        "append_keywords": ["for adults", "lifestyle"],
        "restrict": "none",
    },
    "40-49": {
        "append_keywords": ["professional", "mature content"],
        "restrict": "none",
    },
    "50-59": {
        "append_keywords": ["classic", "nostalgic"],
        "restrict": "none",
    },
    "60+": {
        "append_keywords": ["relaxing", "classic", "nostalgic"],
        "restrict": "none",
    }
}

class EmotionAnalysis:
    def __init__(self):
        # Create a folder named 'known_faces' if it doesn't exist
        if not os.path.exists('known_faces'):
            os.makedirs('known_faces')
            print("Created 'known_faces' directory")
            
        # Load known faces and their names and estimated age ranges
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_faces_age_ranges = {}
        self.known_faces_dir = 'known_faces'
        
        self.emotion_detector = FER()
        self.load_known_faces()
        
    def get_age_range(self, age):
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
            
    def load_known_faces(self):
        """Load known faces from the directory"""
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_faces_age_ranges = {}
        
        print("Loading known faces...")
        
        for filename in os.listdir(self.known_faces_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(self.known_faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                if len(face_encodings) > 0:
                    self.known_face_encodings.append(face_encodings[0])
                    name = os.path.splitext(filename)[0]
                    self.known_face_names.append(name)
                    try:
                        age_predictions = DeepFace.analyze(img_path=image_path, actions=['age'], silent=True)
                        if age_predictions and len(age_predictions) > 0:
                            estimated_age = age_predictions[0]['age']
                            self.known_faces_age_ranges[name] = self.get_age_range(estimated_age)
                        else:
                            self.known_faces_age_ranges[name] = "Unknown Age"
                            print(f"Warning: Could not estimate age for {filename}")
                    except Exception as e:
                        self.known_faces_age_ranges[name] = "Unknown Age"
                        print(f"Warning: Age estimation failed for {filename}: {e}")
                else:
                    print(f"Warning: No face found in {filename}")
                    
        print(f"Loaded {len(self.known_face_names)} known faces")

    def add_new_face(self, frame, face_location):
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
                save_path = os.path.join(self.known_faces_dir, f"{name}.jpg")
                cv2.imwrite(save_path, cv2.cvtColor(np.array(face_image_pil), cv2.COLOR_RGB2BGR))
                print(f"Saved new face as {name}.jpg")

                new_image = face_recognition.load_image_file(save_path)
                new_encoding = face_recognition.face_encodings(new_image)
                if len(new_encoding) > 0:
                    self.known_face_encodings.append(new_encoding[0])
                    self.known_face_names.append(name)
                    try:
                        age_predictions = DeepFace.analyze(img_path=save_path, actions=['age'], silent=True)
                        if age_predictions and len(age_predictions) > 0:
                            estimated_age = age_predictions[0]['age']
                            self.known_faces_age_ranges[name] = self.get_age_range(estimated_age)
                        else:
                            self.known_faces_age_ranges[name] = "Unknown Age"
                            print(f"Warning: Could not estimate age for newly saved face {name}")
                    except Exception as e:
                        self.known_faces_age_ranges[name] = "Unknown Age"
                        print(f"Warning: Age estimation failed for newly saved face {name}: {e}")
                else:
                    print(f"Error: No face found in the saved image {name}.jpg")

        root.destroy()
                
    def run_emotion_detection(self, duration=FACE_DETECTION_DURATION):
        """Run face detection and emotion analysis for specified duration"""
        # Initialize webcam
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("Error: Could not open webcam")
            return {}
            
        start_time = time.time()
        emotion_history = {}  # Dictionary to store emotion history for each detected face
        
        print(f"Starting emotion detection for {duration} seconds...")
        
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Break if time exceeded
            if elapsed_time >= duration:
                break

            ret, frame = video_capture.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                continue
                
            # Resize frame for faster processing (optional)
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_index, face_encoding in enumerate(face_encodings):
                # Scale back up face locations
                top, right, bottom, left = [coord * 4 for coord in face_locations[face_index]]
                
                # See if face matches any known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                recognized_age_range = None
                detected_emotion = None
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                    recognized_age_range = self.known_faces_age_ranges.get(name)

                    # Detect emotion for the recognized face
                    face_image = frame[top:bottom, left:right]
                    emotion_result = self.emotion_detector.detect_emotions(face_image)
                    if emotion_result:
                        emotions = emotion_result[0]['emotions']
                        detected_emotion = max(emotions, key=emotions.get)

                        if name not in emotion_history:
                            emotion_history[name] = []
                        emotion_history[name].append(detected_emotion)

                else:
                    self.add_new_face(frame, (top, right, bottom, left))
                    name = "Recognizing..."

                # Draw rectangle and label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Display current emotion above the bounding box
                if detected_emotion:
                    cv2.putText(frame, f"Emotion: {detected_emotion}", (left + 6, top - 10), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)

                # Draw the name and age below the bounding box
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, f"{name}, {recognized_age_range if recognized_age_range else 'Unknown Age'}", 
                           (left + 6, bottom - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            # Show progress
            remaining = int(duration - elapsed_time)
            cv2.putText(frame, f"Time remaining: {remaining}s", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                       
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

        print(f"\n--- Emotion Analysis Complete ---")
        
        # Process the emotion history to get dominant emotions
        dominant_emotions = {}
        for name, emotions in emotion_history.items():
            if emotions:
                emotion_counts = Counter(emotions)
                dominant_emotion = emotion_counts.most_common(1)[0][0]
                dominant_emotions[name] = {
                    'emotion': dominant_emotion,
                    'age_range': self.known_faces_age_ranges.get(name, "Unknown Age")
                }
                
                # Log to database
                city, country = get_geolocation()
                log_emotion_data(name, self.known_faces_age_ranges.get(name, "Unknown Age"), 
                                dominant_emotion,int(time.time()), city, country)
                
                print(f"- {name}: {dominant_emotion} (Age: {self.known_faces_age_ranges.get(name, 'Unknown Age')})")
        
        return dominant_emotions

class YouTubeRecommender:
    def __init__(self, api_key):
        self.api_key = api_key
        try:
            self.youtube = build('youtube', 'v3', developerKey=api_key)
            print("YouTube API initialized successfully")
        except Exception as e:
            print(f"Error initializing YouTube API: {e}")
            self.youtube = None
            
    def validate_api(self):
        """Check if the YouTube API key is valid"""
        if not self.youtube:
            return False
            
        try:
            # Make a simple API call to test the key
            self.youtube.videos().list(part='snippet', id='dQw4w9WgXcQ').execute()
            return True
        except Exception as e:
            print(f"YouTube API key validation failed: {e}")
            return False
        
    def search_videos(self, query, category_id=None, max_results=10, content_restriction=None):
        """Search for videos based on query and optional category"""
        if not self.youtube:
            print("YouTube API not available")
            return []
            
        try:
            search_params = {
                'q': query,
                'type': 'video',
                'part': 'snippet',
                'maxResults': max_results,
                'videoEmbeddable': 'true',
                'videoSyndicated': 'true',
                'videoDuration': 'medium'  # Medium length videos (4-20 minutes)
            }
            
            if category_id:
                search_params['videoCategoryId'] = category_id
                
            if content_restriction == "strict":
                search_params['safeSearch'] = 'strict'
            elif content_restriction == "moderate":
                search_params['safeSearch'] = 'moderate'
                
            search_response = self.youtube.search().list(**search_params).execute()
            
            videos = []
            for item in search_response.get('items', []):
                video_id = item['id']['videoId']
                title = item['snippet']['title']
                description = item['snippet']['description']
                thumbnail = item['snippet']['thumbnails']['medium']['url']
                channel = item['snippet']['channelTitle']
                
                # Get video statistics
                video_response = self.youtube.videos().list(
                    part='statistics,contentDetails',
                    id=video_id
                ).execute()
                
                if video_response['items']:
                    stats = video_response['items'][0]['statistics']
                    duration = video_response['items'][0]['contentDetails']['duration']
                    
                    videos.append({
                        'id': video_id,
                        'title': title,
                        'description': description,
                        'thumbnail': thumbnail,
                        'channel': channel,
                        'views': int(stats.get('viewCount', 0)),
                        'likes': int(stats.get('likeCount', 0)),
                        'duration': self._parse_duration(duration)
                    })
                    
            # Sort by views (popularity)
            videos.sort(key=lambda x: x['views'], reverse=True)
            return videos
            
        except Exception as e:
            print(f"Error searching YouTube: {e}")
            return []
            
    def _parse_duration(self, duration_str):
        """Convert ISO 8601 duration to human-readable format"""
        # Simple parsing of PT#M#S format
        minutes = 0
        seconds = 0
        
        if 'M' in duration_str:
            minutes_part = duration_str.split('M')[0]
            minutes = int(minutes_part.split('PT')[-1])
            
        if 'S' in duration_str:
            seconds_part = duration_str.split('S')[0]
            if 'M' in seconds_part:
                seconds = int(seconds_part.split('M')[-1])
            else:
                seconds = int(seconds_part.split('PT')[-1])
                
        return f"{minutes}:{seconds:02d}"

class RecommendationEngine:
    def __init__(self, youtube_recommender):
        self.youtube = youtube_recommender
        self.cached_recommendations = {}
        
    def get_recommendations_for_user(self, user_name, emotion_data):
        """Generate personalized video recommendations based on emotion data"""
        # Check cache first (for returning users)
        if user_name in self.cached_recommendations:
            cache_time, recommendations = self.cached_recommendations[user_name]
            # Cache valid for 1 hour
            if datetime.now() - cache_time < timedelta(hours=1):
                return recommendations
        
        if not emotion_data:
            print(f"No emotion data available for {user_name}")
            return []
            
        # Extract data
        dominant_emotion = emotion_data['emotion'].lower()
        age_range = emotion_data['age_range']
        
        # Get age-appropriate content settings
        age_settings = AGE_CONTENT_MAPPING.get(age_range, AGE_CONTENT_MAPPING["20-29"])
        
        # Get emotion-based video preferences
        emotion_settings = EMOTION_VIDEO_MAPPING.get(dominant_emotion, EMOTION_VIDEO_MAPPING["neutral"])
        
        # Get user preferences (if any)
        # user_prefs = get_user_preferences(user_name)
        
        # Build search queries
        all_recommendations = []
        
        # Get geolocation for regional content
        city, country = get_geolocation()
        
        # Try each keyword combined with age preferences
        for keyword in emotion_settings['keywords']:
            # Add age-appropriate keywords
            full_query = f"{keyword} {random.choice(age_settings['append_keywords'])}"
            
            # Sometimes add location-specific content (50% chance)
            if country and random.random() > 0.5:
                full_query += f" {country}"
                
            # Search in each relevant category
            for category in emotion_settings['categories']:
                videos = self.youtube.search_videos(
                    query=full_query,
                    category_id=category,
                    max_results=3,
                    content_restriction=age_settings['restrict']
                )
                all_recommendations.extend(videos)
        
        # Remove duplicate videos (by ID)
        unique_recommendations = []
        seen_ids = set()
        
        for video in all_recommendations:
            if video['id'] not in seen_ids:
                seen_ids.add(video['id'])
                unique_recommendations.append(video)
        
        # Sort by views and limit to 20 recommendations
        unique_recommendations.sort(key=lambda x: x['views'], reverse=True)
        final_recommendations = unique_recommendations[:20]
        
        # Cache the recommendations
        self.cached_recommendations[user_name] = (datetime.now(), final_recommendations)
        
        return final_recommendations

class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion-Based Video Recommendation System")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize components
        self.emotion_analyzer = EmotionAnalysis()
        self.youtube_recommender = YouTubeRecommender(YOUTUBE_API_KEY)
        self.recommendation_engine = RecommendationEngine(self.youtube_recommender)
        
        # Store detected emotions
        self.detected_emotions = {}
        self.current_user = None
        
        # Create UI
        self._create_ui()
        
    def _create_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Emotion-Based Video Recommendation", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Create notebook (tabs)
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Tab 1: Face Detection
        face_tab = ttk.Frame(notebook)
        notebook.add(face_tab, text="Face Detection")
        
        # Tab 2: Recommendations
        recommendation_tab = ttk.Frame(notebook)
        notebook.add(recommendation_tab, text="Video Recommendations")
        
        # Configure Face Detection Tab
        self._setup_face_detection_tab(face_tab)
        
        # Configure Recommendations Tab
        self._setup_recommendations_tab(recommendation_tab)
        
    def _setup_face_detection_tab(self, parent):
        # Face detection controls
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(control_frame, text="Duration (seconds):").pack(side=tk.LEFT, padx=5)
        
        self.duration_var = tk.IntVar(value=FACE_DETECTION_DURATION)
        duration_spin = ttk.Spinbox(control_frame, from_=5, to=60, textvariable=self.duration_var, width=5)
        duration_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Start Face Detection", 
                  command=self._run_face_detection).pack(side=tk.LEFT, padx=20)
        
        ttk.Button(control_frame, text="Refresh Known Faces", 
                  command=self._refresh_known_faces).pack(side=tk.LEFT, padx=5)
        
        # Results frame
        self.results_frame = ttk.LabelFrame(parent, text="Detection Results")
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        self.results_text = tk.Text(self.results_frame, height=20, width=80)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def _setup_recommendations_tab(self, parent):
        # User selection frame
        user_frame = ttk.Frame(parent)
        user_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(user_frame, text="Select User:").pack(side=tk.LEFT, padx=5)
        
        self.user_var = tk.StringVar()
        self.user_dropdown = ttk.Combobox(user_frame, textvariable=self.user_var, width=30)
        self.user_dropdown.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(user_frame, text="Get Recommendations", 
                  command=self._load_recommendations).pack(side=tk.LEFT, padx=20)
        
        # Recommendations display frame
        self.recommendations_frame = ttk.Frame(parent)
        self.recommendations_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create canvas with scrollbar for recommendations
        self.canvas = tk.Canvas(self.recommendations_frame)
        scrollbar = ttk.Scrollbar(self.recommendations_frame, orient="vertical", 
                                 command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame inside canvas
        self.inner_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")
        
        # Configure canvas scrolling
        self.inner_frame.bind("<Configure>", 
                             lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._configure_canvas_window)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Update user dropdown
        self._update_user_dropdown()
        
    def _configure_canvas_window(self, event):
        """Adjust the width of the canvas window when canvas is resized"""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
        
    def _refresh_known_faces(self):
        """Refresh the list of known faces"""
        self.emotion_analyzer.load_known_faces()
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Known faces refreshed.\n\n")
        self.results_text.insert(tk.END, f"Loaded {len(self.emotion_analyzer.known_face_names)} faces:\n")
        
        for i, name in enumerate(self.emotion_analyzer.known_face_names):
            age = self.emotion_analyzer.known_faces_age_ranges.get(name, "Unknown Age")
            self.results_text.insert(tk.END, f"{i+1}. {name} - {age}\n")
            
        # Update user dropdown also
        self._update_user_dropdown()
        
    def _run_face_detection(self):
        """Run face detection in a separate thread"""
        duration = self.duration_var.get()
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Starting face detection for {duration} seconds...\n")
        self.results_text.insert(tk.END, "Please look at the camera.\n\n")
        self.root.update()
        
        # Run detection in a separate thread to avoid freezing UI
        threading.Thread(target=self._detection_thread, args=(duration,), daemon=True).start()
        
    def _detection_thread(self, duration):
        """Thread for running face detection"""
        # Run detection
        detected_emotions = self.emotion_analyzer.run_emotion_detection(duration)
        self.detected_emotions = detected_emotions
        
        # Update UI in main thread
        self.root.after(0, lambda: self._update_detection_results(detected_emotions))
        
    def _update_detection_results(self, results):
        """Update UI with detection results"""
        self.results_text.delete(1.0, tk.END)
        
        if not results:
            self.results_text.insert(tk.END, "No faces detected during the session.\n")
            return
            
        self.results_text.insert(tk.END, "Detection Results:\n\n")
        
        for name, data in results.items():
            self.results_text.insert(tk.END, f"Name: {name}\n")
            self.results_text.insert(tk.END, f"Age Range: {data['age_range']}\n")
            self.results_text.insert(tk.END, f"Dominant Emotion: {data['emotion']}\n\n")
            
        self.results_text.insert(tk.END, "\nReady to generate video recommendations. "
                                        "Please go to the Recommendations tab.")
                                        
        # Update user dropdown in recommendations tab
        self._update_user_dropdown()
        
    def _update_user_dropdown(self):
        """Update the user dropdown with known faces"""
        current_selection = self.user_var.get()
        
        # Get users from known faces
        users = self.emotion_analyzer.known_face_names
        
        self.user_dropdown['values'] = users
        
        # Try to keep current selection if possible
        if current_selection in users:
            self.user_var.set(current_selection)
        elif users:
            self.user_var.set(users[0])
        else:
            self.user_var.set("")
            
    def _load_recommendations(self):
        """Load video recommendations for selected user"""
        user_name = self.user_var.get()
        if not user_name:
            messagebox.showinfo("Selection Required", "Please select a user.")
            return
            
        self.current_user = user_name
        
        # Clear existing recommendations
        for widget in self.inner_frame.winfo_children():
            widget.destroy()
            
        # Check if we have emotion data for this user
        user_emotion = None
        if user_name in self.detected_emotions:
            user_emotion = self.detected_emotions[user_name]
        else:
            # Query from database (most recent emotion)
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT emotion, age_range FROM emotion_logs 
                WHERE name = ? ORDER BY timestamp DESC LIMIT 1
            """, (user_name,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                user_emotion = {'emotion': result[0], 'age_range': result[1]}
                
        if not user_emotion:
            messagebox.showinfo("No Data", 
                              f"No emotion data available for {user_name}. "
                              f"Please run face detection first.")
            return
            
        # Update status
        self.status_var.set(f"Loading recommendations for {user_name}...")
        self.root.update()
        
        # Get recommendations in a separate thread
        threading.Thread(target=self._fetch_recommendations_thread, 
                        args=(user_name, user_emotion), daemon=True).start()
    

    def _fetch_recommendations_thread(self, user_name, emotion_data):
        """Background thread to fetch recommendations"""
        recommendations = self.recommendation_engine.get_recommendations_for_user(user_name, emotion_data)
        
        # Update UI in main thread
        self.root.after(0, lambda: self._display_recommendations(recommendations, user_name, emotion_data))
        
    def _display_recommendations(self, recommendations, user_name, emotion_data):
        """Display video recommendations in the UI"""
        # Clear existing content first
        for widget in self.inner_frame.winfo_children():
            widget.destroy()
            
        # Display user info
        ttk.Label(self.inner_frame, text=f"Recommendations for: {user_name}", 
                font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=5)
        ttk.Label(self.inner_frame, text=f"Detected Emotion: {emotion_data['emotion']}", 
                font=("Arial", 10)).pack(anchor="w", padx=10, pady=2)
        ttk.Label(self.inner_frame, text=f"Age Range: {emotion_data['age_range']}", 
                font=("Arial", 10)).pack(anchor="w", padx=10, pady=2)
        
        ttk.Separator(self.inner_frame, orient="horizontal").pack(fill="x", padx=10, pady=10)
        
        if not recommendations:
            ttk.Label(self.inner_frame, text="No recommendations found.", 
                    font=("Arial", 10, "italic")).pack(pady=20)
            self.status_var.set("No recommendations found")
            return
            
        # Display each recommendation
        for i, video in enumerate(recommendations):
            # Create frame for video
            video_frame = ttk.Frame(self.inner_frame)
            video_frame.pack(fill="x", padx=10, pady=5)
            
            # Load thumbnail
            try:
                response = requests.get(video['thumbnail'])
                img_data = response.content
                img = Image.open(BytesIO(img_data))
                img = img.resize((120, 90), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Keep a reference to prevent garbage collection
                video_frame.thumbnail = photo
                
                thumbnail_label = ttk.Label(video_frame, image=photo)
                thumbnail_label.pack(side="left", padx=5)
            except Exception as e:
                print(f"Error loading thumbnail: {e}")
                thumbnail_label = ttk.Label(video_frame, text="[Thumbnail]", width=15, relief="ridge")
                thumbnail_label.pack(side="left", padx=5)
            
            # Video information
            info_frame = ttk.Frame(video_frame)
            info_frame.pack(side="left", fill="both", expand=True, padx=5)
            
            ttk.Label(info_frame, text=video['title'], 
                    font=("Arial", 10, "bold"), wraplength=400).pack(anchor="w")
            ttk.Label(info_frame, text=f"Channel: {video['channel']}", 
                    font=("Arial", 9)).pack(anchor="w")
            ttk.Label(info_frame, text=f"Views: {video['views']:,} • Duration: {video['duration']}", 
                    font=("Arial", 9)).pack(anchor="w")
            
            # Watch button
            video_id = video['id']
            ttk.Button(video_frame, text="Watch", 
                    command=lambda vid=video_id: self._open_youtube_video(vid)).pack(side="right", padx=10)
        
        self.status_var.set(f"Loaded {len(recommendations)} recommendations for {user_name}")
        
    def _open_youtube_video(self, video_id):
        """Open YouTube video in browser"""
        url = f"https://www.youtube.com/watch?v={video_id}"
        webbrowser.open_new(url)

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()