import numpy as np
from datetime import datetime
import sqlite3

class UserFeatureExtractor:
    """Extract and process user features for the neural network"""
    
    def __init__(self, database_path='user_data.db'):
        """Initialize user feature extractor
        
        Args:
            database_path: Path to SQLite database
        """
        self.db_path = database_path
        
        # Create database connection
        self.conn = None
        self.create_tables()
        
    def create_tables(self):
        """Create necessary database tables if they don't exist"""
        try:
            # Connect to database
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            # Create emotion logs table if it doesn't exist
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
            
            # Create user preferences table for storing neural network data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    feature_vector TEXT,  -- JSON string of feature vector values
                    last_updated INTEGER
                )
            """)
            
            # Create video interaction table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS video_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    video_id TEXT,
                    emotion TEXT,
                    interaction_type TEXT,  -- "watched", "liked", "clicked", etc.
                    timestamp INTEGER
                )
            """)
            
            # Create a table to store the trained categories for each user
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_categories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    category TEXT,
                    score REAL,
                    timestamp INTEGER
                )
            """)
            
            # Create a table to store the trained keywords for each user
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_keywords (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    keyword TEXT,
                    score REAL,
                    timestamp INTEGER
                )
            """)
            
            self.conn.commit()
            
        except Exception as e:
            print(f"Error creating tables: {e}")
            if self.conn:
                self.conn.close()
                self.conn = None
        
    def extract_user_features(self, name):
        """Extract user features from database
        
        Args:
            name: User name
            
        Returns:
            feature_vector: NumPy array of user features
        """
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path)
            
        cursor = self.conn.cursor()
        
        # Get user's emotion history
        cursor.execute("""
            SELECT emotion, COUNT(*) as count FROM emotion_logs 
            WHERE name = ? GROUP BY emotion ORDER BY count DESC
        """, (name,))
        emotion_counts = cursor.fetchall()
        
        # Get user's video interactions
        cursor.execute("""
            SELECT video_id, emotion FROM video_interactions 
            WHERE name = ? ORDER BY timestamp DESC LIMIT 20
        """, (name,))
        interactions = cursor.fetchall()
        
        # Get user categories
        cursor.execute("""
            SELECT category, score FROM user_categories 
            WHERE name = ? ORDER BY score DESC
        """, (name,))
        categories = cursor.fetchall()
        
        # Get user keywords
        cursor.execute("""
            SELECT keyword, score FROM user_keywords 
            WHERE name = ? ORDER BY score DESC
        """, (name,))
        keywords = cursor.fetchall()
        
        # Calculate emotional stability (variance of emotions over time)
        cursor.execute("""
            SELECT emotion, timestamp FROM emotion_logs 
            WHERE name = ? ORDER BY timestamp
        """, (name,))
        emotion_history = cursor.fetchall()
        
        # Build feature vector
        features = {}
        
        # Add emotion distribution
        for emotion, count in emotion_counts:
            features[f'emotion_{emotion}'] = count
            
        # Add interaction information
        if interactions:
            features['has_interactions'] = 1
            
            # Most recent emotions from interactions
            recent_emotions = [interaction[1] for interaction in interactions[:5]]
            for i, emotion in enumerate(recent_emotions):
                features[f'recent_emotion_{i}'] = emotion
                
            # Calculate engagement level
            features['engagement_level'] = len(interactions) / 20.0  # Normalized to 0-1
        else:
            features['has_interactions'] = 0
            
        # Add category preferences
        if categories:
            for category, score in categories[:5]:  # Top 5 categories
                features[f'category_{category}'] = score
                
        # Add keyword preferences
        if keywords:
            for keyword, score in keywords[:5]:  # Top 5 keywords
                features[f'keyword_{keyword}'] = score
                
        # Convert to numpy array
        feature_vector = np.array(list(features.values()))
        
        return feature_vector
        
    def log_user_video_interaction(self, name, video_id, emotion, interaction_type='watched'):
        """Log user-video interaction
        
        Args:
            name: User name
            video_id: YouTube video ID
            emotion: User emotion at time of interaction
            interaction_type: Type of interaction (watched, liked, etc.)
        """
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path)
            
        cursor = self.conn.cursor()
        
        timestamp = int(datetime.now().timestamp())
        
        cursor.execute("""
            INSERT INTO video_interactions 
            (name, video_id, emotion, interaction_type, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (name, video_id, emotion, interaction_type, timestamp))
        
        self.conn.commit()
        
    def update_user_categories(self, name, categories, scores):
        """Update user category preferences
        
        Args:
            name: User name
            categories: List of categories
            scores: List of scores (0-1) for each category
        """
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path)
            
        cursor = self.conn.cursor()
        
        timestamp = int(datetime.now().timestamp())
        
        # Delete old categories for this user
        cursor.execute("DELETE FROM user_categories WHERE name = ?", (name,))
        
        # Insert new categories
        for category, score in zip(categories, scores):
            cursor.execute("""
                INSERT INTO user_categories
                (name, category, score, timestamp)
                VALUES (?, ?, ?, ?)
            """, (name, category, score, timestamp))
            
        self.conn.commit()
        
    def update_user_keywords(self, name, keywords, scores):
        """Update user keyword preferences
        
        Args:
            name: User name
            keywords: List of keywords
            scores: List of scores (0-1) for each keyword
        """
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path)
            
        cursor = self.conn.cursor()
        
        timestamp = int(datetime.now().timestamp())
        
        # Delete old keywords for this user
        cursor.execute("DELETE FROM user_keywords WHERE name = ?", (name,))
        
        # Insert new keywords
        for keyword, score in zip(keywords, scores):
            cursor.execute("""
                INSERT INTO user_keywords
                (name, keyword, score, timestamp)
                VALUES (?, ?, ?, ?)
            """, (name, keyword, score, timestamp))
            
        self.conn.commit()
