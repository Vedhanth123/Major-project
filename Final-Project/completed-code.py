import streamlit as st
import cv2
import face_recognition
import os
import numpy as np
from fer import FER
from collections import Counter
import time
import sqlite3
import requests
from io import BytesIO
from PIL import Image
from googleapiclient.discovery import build
import random
from datetime import datetime, timedelta
import threading
import google.generativeai as genai  # Import Google's Gemini API

# Configuration
YOUTUBE_API_KEY = "AIzaSyAVer6TDzEBXthg6l4LtxJSMW6ybHGqB8c"  # Replace with your actual API key
GEMINI_API_KEY = "AIzaSyCGkhZrVWs94XBT0amuoihdE8wM9P1Nb-I"  # Replace with your Gemini API key
DATABASE_PATH = "user_data.db" 
MAX_RESULTS = 10
FACE_DETECTION_DURATION = 8  # Default seconds to run face detection

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Create directory for known faces if it doesn't exist
if not os.path.exists('known_faces'):
    os.makedirs('known_faces')
    st.write("Created 'known_faces' directory")

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

# Emotion descriptions for Gemini context
EMOTION_DESCRIPTIONS = {
    "happy": "You're in a happy mood. You might enjoy content that amplifies your positive feelings, makes you laugh, or celebrates joyful moments.",
    "sad": "You seem to be feeling sad. You might benefit from content that provides comfort, gentle uplift, or helps process emotions.",
    "angry": "You appear to be feeling angry. You might benefit from content that helps you calm down, relax, or redirect your focus.",
    "fear": "You seem to be experiencing fear or anxiety. You might benefit from content that provides reassurance, calm, or positive distraction.",
    "surprise": "You look surprised. You might enjoy content that further stimulates your curiosity or shows you more amazing things.",
    "neutral": "You appear to be in a neutral mood. You might enjoy informative content that engages your mind.",
    "disgust": "You seem to be feeling disgusted. You might benefit from content that provides pleasant visual relief or positive sensory experiences."
}

# Function to get geolocation (simplified version)
def get_geolocation():
    # In a real app, this would use IP geolocation or user input
    # For simplicity, we'll return default values
    return "Unknown City", "Unknown Country"

# Function to log emotion data to database
def log_emotion_data(name, emotion, timestamp, city, country):
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS emotion_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            emotion TEXT,
            timestamp INTEGER,
            city TEXT,
            country TEXT
        )
        ''')
        
        # Insert data
        cursor.execute('''
        INSERT INTO emotion_logs (name, emotion, timestamp, city, country)
        VALUES (?, ?, ?, ?, ?)
        ''', (name, emotion, timestamp, city, country))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database error: {e}")
        return False

class EmotionAnalysis:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_faces_dir = 'known_faces'
        self.emotion_detector = FER()
        self.load_known_faces()
        
    def load_known_faces(self):
        """Load known faces from the directory"""
        self.known_face_encodings = []
        self.known_face_names = []
        
        for filename in os.listdir(self.known_faces_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(self.known_faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                if len(face_encodings) > 0:
                    self.known_face_encodings.append(face_encodings[0])
                    name = os.path.splitext(filename)[0]
                    self.known_face_names.append(name)
                else:
                    st.warning(f"No face found in {filename}")
                    
        return len(self.known_face_names)

    def add_new_face(self, frame, face_location, name):
        """Add a new face with the given name"""
        if not name or name.strip() == "":
            return False
            
        name = name.strip()
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]
        
        save_path = os.path.join(self.known_faces_dir, f"{name}.jpg")
        cv2.imwrite(save_path, face_image)
        
        new_image = face_recognition.load_image_file(save_path)
        new_encoding = face_recognition.face_encodings(new_image)
        if len(new_encoding) > 0:
            self.known_face_encodings.append(new_encoding[0])
            self.known_face_names.append(name)
            return True
        else:
            os.remove(save_path)  # Remove the file if face detection failed
            return False
                
    def run_emotion_detection(self, stframe, status_text, progress_bar, duration=FACE_DETECTION_DURATION):
        """Run face detection and emotion analysis for specified duration"""
        # Initialize webcam
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            status_text.error("Error: Could not open webcam")
            return {}
            
        start_time = time.time()
        emotion_history = {}  # Dictionary to store emotion history for each detected face
        
        status_text.info(f"Starting emotion detection for {duration} seconds...")
        
        # For new faces that need names
        pending_faces = {}
        
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
                
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            # Process each detected face
            for face_index, face_encoding in enumerate(face_encodings):
                # Scale back up face locations
                top, right, bottom, left = [coord * 4 for coord in face_locations[face_index]]
                
                # See if face matches any known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                detected_emotion = None
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]

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
                    # Store face location for naming later
                    face_img = frame[top:bottom, left:right].copy()
                    face_key = f"face_{face_index}_{int(time.time())}"
                    pending_faces[face_key] = {
                        "image": face_img,
                        "location": (top, right, bottom, left)
                    }
                    name = "Unknown"

                # Draw rectangle and label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Display current emotion above the bounding box
                if detected_emotion:
                    cv2.putText(frame, f"Emotion: {detected_emotion}", (left + 6, top - 10), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)

                # Draw the name below the bounding box
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 10), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            # Show progress
            cv2.putText(frame, f"Time remaining: {int(remaining_time)}s", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                       
            # Display the frame in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")

        video_capture.release()
        
        # Process the emotion history to get dominant emotions
        dominant_emotions = {}
        for name, emotions in emotion_history.items():
            if emotions:
                emotion_counts = Counter(emotions)
                dominant_emotion = emotion_counts.most_common(1)[0][0]
                dominant_emotions[name] = {
                    'emotion': dominant_emotion
                }
                
                # Log to database
                city, country = get_geolocation()
                log_emotion_data(name, dominant_emotion, int(time.time()), city, country)
        
        return dominant_emotions, pending_faces

class GeminiExplainer:
    def __init__(self, api_key):
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-pro')
            self.available = True
        except Exception as e:
            st.error(f"Error initializing Gemini: {e}")
            self.available = False
        
        # Cache for explanations to avoid duplicate API calls
        self.explanation_cache = {}
            
    def explain_recommendation(self, video, user_name, emotion):
        """Generate explanation for why this video is recommended"""
        
        # Create cache key based on video ID and emotion
        cache_key = f"{video['id']}_{emotion}"
        
        # Return cached explanation if available
        if cache_key in self.explanation_cache:
            return self.explanation_cache[cache_key]
        
        if not self.available:
            return "Video recommended based on your current emotion."
            
        try:
            # Create prompt for Gemini
            prompt = f"""
            You are an AI assistant helping explain video recommendations to users.
            
            USER: {user_name}
            DETECTED EMOTION: {emotion}
            EMOTION CONTEXT: {EMOTION_DESCRIPTIONS.get(emotion, "You might enjoy content tailored to your current mood.")}
            
            VIDEO DETAILS:
            Title: {video['title']}
            Channel: {video['channel']}
            Description: {video['description'][:200]}...
            
            Explain in 1-2 brief, personalized sentences why this video might benefit the user given their current emotional state.
            Keep your explanation friendly, supportive, and under 30 words.
            Start directly with the explanation without phrases like "This video is recommended because".
            """
            
            # Generate explanation from Gemini
            response = self.model.generate_content(prompt)
            explanation = response.text.strip()
            
            # Clean up explanation
            explanation = explanation.replace('"', '')
            if len(explanation) > 100:
                explanation = explanation[:97] + "..."
                
            # Cache the explanation
            self.explanation_cache[cache_key] = explanation
            
            return explanation
            
        except Exception as e:
            print(f"Gemini error: {e}")
            return "Video recommended based on your current emotion."

class YouTubeRecommender:
    def __init__(self, api_key):
        self.api_key = api_key
        try:
            self.youtube = build('youtube', 'v3', developerKey=api_key)
        except Exception as e:
            st.error(f"Error initializing YouTube API: {e}")
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
            st.error(f"YouTube API key validation failed: {e}")
            return False
        
    def search_videos(self, query, category_id=None, max_results=10):
        """Search for videos based on query and optional category"""
        if not self.youtube:
            st.error("YouTube API not available")
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
            st.error(f"Error searching YouTube: {e}")
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
    def __init__(self, youtube_recommender, gemini_explainer):
        self.youtube = youtube_recommender
        self.gemini = gemini_explainer
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
            st.warning(f"No emotion data available for {user_name}")
            return []
            
        # Extract data
        dominant_emotion = emotion_data['emotion'].lower()
        
        # Get emotion-based video preferences
        emotion_settings = EMOTION_VIDEO_MAPPING.get(dominant_emotion, EMOTION_VIDEO_MAPPING["neutral"])
        
        # Build search queries
        all_recommendations = []
        
        # Get geolocation for regional content
        city, country = get_geolocation()
        
        # Try each keyword
        for keyword in emotion_settings['keywords']:
            full_query = keyword
            
            # Sometimes add location-specific content (50% chance)
            if country and country != "Unknown Country" and random.random() > 0.5:
                full_query += f" {country}"
                
            # Search in each relevant category
            for category in emotion_settings['categories']:
                videos = self.youtube.search_videos(
                    query=full_query,
                    category_id=category,
                    max_results=3
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
        
        # Add explanations to each recommendation
        for video in final_recommendations:
            video['explanation'] = self.gemini.explain_recommendation(
                video, 
                user_name, 
                dominant_emotion
            )
        
        # Cache the recommendations
        self.cached_recommendations[user_name] = (datetime.now(), final_recommendations)
        
        return final_recommendations

def main():
    st.set_page_config(
        page_title="Emotion-Based Video Recommendation",
        page_icon="📺",
        layout="wide",
    )
    
    # Initialize session state for storing data between reruns
    if 'emotion_analyzer' not in st.session_state:
        st.session_state.emotion_analyzer = EmotionAnalysis()
    
    if 'youtube_recommender' not in st.session_state:
        st.session_state.youtube_recommender = YouTubeRecommender(YOUTUBE_API_KEY)
    
    if 'gemini_explainer' not in st.session_state:
        st.session_state.gemini_explainer = GeminiExplainer(GEMINI_API_KEY)
    
    if 'recommendation_engine' not in st.session_state:
        st.session_state.recommendation_engine = RecommendationEngine(
            st.session_state.youtube_recommender,
            st.session_state.gemini_explainer
        )
    
    if 'detected_emotions' not in st.session_state:
        st.session_state.detected_emotions = {}
    
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    
    if 'pending_faces' not in st.session_state:
        st.session_state.pending_faces = {}
        
    # App title and description
    st.title("Emotion-Based Video Recommendation")
    st.markdown("This application detects your facial emotions and recommends YouTube videos based on your mood.")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Face Detection", "Video Recommendations"])
    
    # Face Detection Tab
    with tab1:
        st.header("Face Detection")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("Controls")
            duration = st.slider("Detection Duration (seconds)", 
                               min_value=5, max_value=60, value=FACE_DETECTION_DURATION)
            
            start_detection = st.button("Start Face Detection", use_container_width=True)
            refresh_faces = st.button("Refresh Known Faces", use_container_width=True)
            
            if refresh_faces:
                num_faces = st.session_state.emotion_analyzer.load_known_faces()
                st.success(f"Loaded {num_faces} known faces")
            
            # Display known faces
            st.subheader("Known Faces")
            if st.session_state.emotion_analyzer.known_face_names:
                for name in st.session_state.emotion_analyzer.known_face_names:
                    st.text(f"• {name}")
            else:
                st.info("No known faces yet")
        
        with col1:
            # Video feed and results
            video_placeholder = st.empty()
            status_text = st.empty()
            progress_bar = st.progress(0)
            results_area = st.expander("Detection Results", expanded=True)
            
            # Process any pending faces that need names
            if st.session_state.pending_faces:
                st.subheader("Unidentified Faces")
                st.write("Please name these detected faces:")
                
                cols = st.columns(3)
                faces_to_remove = []
                
                for i, (face_key, face_data) in enumerate(st.session_state.pending_faces.items()):
                    col_idx = i % 3
                    with cols[col_idx]:
                        st.image(face_data["image"], caption=f"Face #{i+1}", width=150)
                        face_name = st.text_input(f"Name for Face #{i+1}", key=f"face_name_{face_key}")
                        
                        if st.button(f"Save Face #{i+1}", key=f"save_face_{face_key}"):
                            if face_name and face_name.strip():
                                success = st.session_state.emotion_analyzer.add_new_face(
                                    cv2.cvtColor(face_data["image"], cv2.COLOR_RGB2BGR), 
                                    (0, face_data["image"].shape[1], face_data["image"].shape[0], 0),
                                    face_name
                                )
                                if success:
                                    st.success(f"Face saved as {face_name}")
                                    faces_to_remove.append(face_key)
                                else:
                                    st.error("Failed to save face")
                            else:
                                st.warning("Please enter a name")
                
                # Remove processed faces
                for key in faces_to_remove:
                    del st.session_state.pending_faces[key]
                
                if faces_to_remove:
                    st.experimental_rerun()
            
            if start_detection:
                with results_area:
                    st.info("Detection started. Please look at the camera.")
                    
                # Reset state
                st.session_state.detected_emotions = {}
                
                # Run detection
                detected_emotions, pending_faces = st.session_state.emotion_analyzer.run_emotion_detection(
                    video_placeholder, status_text, progress_bar, duration
                )
                
                # Store results
                st.session_state.detected_emotions = detected_emotions
                st.session_state.pending_faces = pending_faces
                
                # Show results
                with results_area:
                    if detected_emotions:
                        st.success("Detection Complete!")
                        st.subheader("Detection Results:")
                        
                        for name, data in detected_emotions.items():
                            st.markdown(f"**Name:** {name}")
                            st.markdown(f"**Dominant Emotion:** {data['emotion']}")
                            st.markdown("---")
                            
                        st.info("You can now go to the 'Video Recommendations' tab to get personalized recommendations.")
                    else:
                        st.warning("No faces with emotions were detected. Please try again.")
    
    # Video Recommendations Tab
    with tab2:
        st.header("Video Recommendations")
        
        # User selection
        users = st.session_state.emotion_analyzer.known_face_names
        if users:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_user = st.selectbox(
                    "Select User",
                    options=users,
                    index=0 if st.session_state.current_user not in users else users.index(st.session_state.current_user)
                )
            
            with col2:
                get_recommendations = st.button("Get Recommendations", use_container_width=True)
            
            if get_recommendations:
                st.session_state.current_user = selected_user
                
                # Check if we have emotion data for this user
                user_emotion = None
                if selected_user in st.session_state.detected_emotions:
                    user_emotion = st.session_state.detected_emotions[selected_user]
                else:
                    # Query from database (most recent emotion)
                    try:
                        conn = sqlite3.connect(DATABASE_PATH)
                        cursor = conn.cursor()
                        cursor.execute("""
                            CREATE TABLE IF NOT EXISTS emotion_logs (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                name TEXT,
                                emotion TEXT,
                                timestamp INTEGER,
                                city TEXT,
                                country TEXT
                            )
                        """)
                        cursor.execute("""
                            SELECT emotion FROM emotion_logs 
                            WHERE name = ? ORDER BY timestamp DESC LIMIT 1
                        """, (selected_user,))
                        result = cursor.fetchone()
                        conn.close()
                        
                        if result:
                            user_emotion = {'emotion': result[0]}
                    except Exception as e:
                        st.error(f"Database error: {e}")
                        
                if not user_emotion:
                    st.warning(f"No emotion data available for {selected_user}. Please run face detection first.")
                else:
                    with st.spinner(f"Loading recommendations for {selected_user}..."):
                        recommendations = st.session_state.recommendation_engine.get_recommendations_for_user(
                            selected_user, user_emotion
                        )
                        
                        if recommendations:
                            st.success(f"Found {len(recommendations)} recommendations based on {user_emotion['emotion']} emotion")
                            
                            # Display user info
                            st.markdown(f"### Recommendations for: {selected_user}")
                            st.markdown(f"**Detected Emotion:** {user_emotion['emotion']}")
                            st.markdown(f"**Recommendation Focus:** {EMOTION_DESCRIPTIONS.get(user_emotion['emotion'].lower(), '')}")
                            st.markdown("---")
                            
                            # Display recommendations in a grid
                            cols_per_row = 2
                            for i in range(0, len(recommendations), cols_per_row):
                                cols = st.columns(cols_per_row)
                                for j in range(cols_per_row):
                                    if i + j < len(recommendations):
                                        video = recommendations[i + j]
                                        with cols[j]:
                                            st.markdown(f"#### {video['title']}")
                                            st.image(video['thumbnail'])
                                            
                                            # Display the Gemini-generated explanation in a highlighted box
                                            st.markdown(f"""
                                            <div style="background-color: #f0f7fb; border-left: 5px solid #2196F3; padding: 10px; margin-bottom: 10px;">
                                                <strong>Why this might help:</strong> {video['explanation']}
                                            </div>
                                            """, unsafe_allow_html=True)
                                            
                                            st.markdown(f"**Channel:** {video['channel']}")
                                            st.markdown(f"**Views:** {video['views']:,} • **Duration:** {video['duration']}")
                                            st.markdown(f"**Description:** {video['description'][:100]}...")
                                            
                                            # YouTube embed or link
                                            video_id = video['id']
                                            st.markdown(f'''
                                            <div style="text-align: center;">
                                                <a href="https://www.youtube.com/watch?v={video_id}" target="_blank">
                                                    <button style="
                                                        background-color: #FF0000;
                                                        color: white;
                                                        border: none;
                                                        border-radius: 4px;
                                                        padding: 10px 16px;
                                                        font-size: 16px;
                                                        cursor: pointer;
                                                        display: inline-flex;
                                                        align-items: center;
                                                        margin: 10px 0;">
                                                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="white" style="margin-right: 8px;">
                                                            <path d="M10,16.5V7.5L16,12M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2Z" />
                                                        </svg>
                                                        Watch on YouTube
                                                    </button>
                                                </a>
                                            </div>
                                            ''', unsafe_allow_html=True)
                                            
                                            st.markdown("---")
                        else:
                            st.error("Failed to get recommendations. Please try again.")
        else:
            st.warning("No known users found. Please add users via Face Detection first.")

if __name__ == "__main__":
    main()