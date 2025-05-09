import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import sqlite3

class TrainingDataGenerator:
    """Generate training data for the emotion-video neural network"""
    
    def __init__(self, database_path='user_data.db', 
                 emotion_video_mapping=None):
        """Initialize the training data generator
        
        Args:
            database_path: Path to SQLite database
            emotion_video_mapping: Dictionary mapping emotions to video preferences
        """
        self.db_path = database_path
        self.emotion_video_mapping = emotion_video_mapping or {}
        
    def generate_synthetic_data(self, num_samples=1000):
        """Generate synthetic training data
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            DataFrame with training data
        """
        # Define emotions
        emotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'neutral', 'disgust']
        
        # Define categories
        categories = [
            '1', '2', '10', '15', '17',  # Film & Animation, Autos, Music, Pets & Animals, Sports
            '18', '19', '20', '21', '22',  # Short Movies, Travel, Gaming, Videoblogging, People & Blogs
            '23', '24', '25', '26', '27',  # Comedy, Entertainment, News & Politics, Howto & Style, Education
            '28', '29', '30', '31', '32',  # Science & Tech, Nonprofits & Activism, Movies, Anime, Action/Adventure
            '33', '34', '35', '36', '37',  # Classics, Comedy, Documentary, Drama, Family
            '38', '39', '40', '41', '42',  # Foreign, Horror, Sci-Fi/Fantasy, Thriller, Shorts
            '43', '44'  # Shows, Trailers
        ]
        
        # Define keywords
        all_keywords = [
            'relaxing music', 'calming videos', 'motivational speeches', 'inspirational stories',
            'uplifting music', 'comedy videos', 'funny moments', 'feel good content',
            'calming music', 'meditation videos', 'relaxation techniques', 'nature sounds',
            'soothing music', 'positive affirmations', 'calming content', 'guided relaxation',
            'amazing facts', 'incredible discoveries', 'wow moments', 'mind blowing',
            'interesting documentaries', 'educational content', 'informative videos', 'how-to guides',
            'satisfying videos', 'clean organization', 'aesthetic content', 'art videos',
            'history', 'animals', 'science', 'space', 'technology', 'cooking', 'exercise',
            'dance', 'painting', 'music', 'gaming', 'reviews', 'tutorials', 'DIY', 'crafts'
        ]
        
        # Generate user IDs
        user_ids = [f'user_{i}' for i in range(100)]
        
        # Generate data
        data = []
        
        for _ in range(num_samples):
            # Select random user
            user_id = random.choice(user_ids)
            
            # Select random emotion
            emotion = random.choice(emotions)
            
            # Generate user features
            # For simplicity, we'll use random values here
            # In a real implementation, you would extract these from user behavior
            features = np.random.normal(0, 1, 10)  # 10 random features
            
            # Select categories and keywords based on emotion (with noise)
            if self.emotion_video_mapping and emotion in self.emotion_video_mapping:
                # Use the mapping with some randomness
                base_categories = self.emotion_video_mapping[emotion].get('categories', [])
                base_keywords = self.emotion_video_mapping[emotion].get('keywords', [])
                
                # Add some randomness to categories
                preferred_categories = base_categories.copy()
                for _ in range(random.randint(0, 2)):
                    preferred_categories.append(random.choice(categories))
                    
                # Add some randomness to keywords
                preferred_keywords = base_keywords.copy()
                for _ in range(random.randint(0, 3)):
                    preferred_keywords.append(random.choice(all_keywords))
            else:
                # Completely random preferences
                num_categories = random.randint(1, 4)
                preferred_categories = random.sample(categories, num_categories)
                
                num_keywords = random.randint(2, 6)
                preferred_keywords = random.sample(all_keywords, num_keywords)
            
            # Add to dataset
            data.append({
                'user_id': user_id,
                'emotion': emotion,
                'features': features,
                'watch_history': [],  # Placeholder for real watch history
                'preferred_categories': preferred_categories,
                'preferred_keywords': preferred_keywords
            })
        
        return pd.DataFrame(data)
    
    def generate_from_database(self):
        """Generate training data from database
        
        Returns:
            DataFrame with training data from real user interactions
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all users
            cursor.execute("SELECT DISTINCT name FROM emotion_logs")
            users = [row[0] for row in cursor.fetchall()]
            
            data = []
            
            for user in users:
                # Get user's emotions
                cursor.execute("""
                    SELECT emotion, timestamp FROM emotion_logs
                    WHERE name = ? ORDER BY timestamp DESC
                """, (user,))
                emotions = cursor.fetchall()
                
                if not emotions:
                    continue
                    
                # Get user's video interactions
                cursor.execute("""
                    SELECT video_id, emotion, interaction_type, timestamp 
                    FROM video_interactions
                    WHERE name = ? ORDER BY timestamp DESC
                """, (user,))
                interactions = cursor.fetchall()
                
                # For each emotion instance, create a training sample
                for emotion, timestamp in emotions:
                    # Get video interactions after this emotion detection
                    relevant_interactions = [
                        i for i in interactions 
                        if i[3] >= timestamp and i[3] <= timestamp + 3600  # Within 1 hour
                    ]
                    
                    if not relevant_interactions:
                        continue
                        
                    # Extract watch history
                    watch_history = [i[0] for i in relevant_interactions]
                    
                    # Extract preferred categories and keywords
                    # In a real implementation, you'd get these from the YouTube API
                    # Here we'll use placeholders
                    preferred_categories = []
                    preferred_keywords = []
                    
                    if self.emotion_video_mapping and emotion in self.emotion_video_mapping:
                        preferred_categories = self.emotion_video_mapping[emotion].get('categories', [])
                        preferred_keywords = self.emotion_video_mapping[emotion].get('keywords', [])
                    
                    # Generate mock features (would be real user features in production)
                    features = np.random.normal(0, 1, 10)
                    
                    # Add to dataset
                    data.append({
                        'user_id': user,
                        'emotion': emotion,
                        'features': features,
                        'watch_history': watch_history,
                        'preferred_categories': preferred_categories,
                        'preferred_keywords': preferred_keywords
                    })
            
            conn.close()
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"Error generating data from database: {e}")
            return pd.DataFrame()
    
    def get_combined_dataset(self, synthetic_ratio=0.5, min_samples=500):
        """Get a combined dataset of real and synthetic data
        
        Args:
            synthetic_ratio: Ratio of synthetic to real data (0-1)
            min_samples: Minimum number of total samples
            
        Returns:
            DataFrame with combined training data
        """
        # Get real data
        real_data = self.generate_from_database()
        
        # Calculate how many synthetic samples we need
        num_real = len(real_data)
        num_synthetic = max(
            min_samples - num_real,  # At least min_samples total
            int(num_real * synthetic_ratio / (1 - synthetic_ratio))  # Maintain synthetic_ratio
        )
        
        # Generate synthetic data
        synthetic_data = self.generate_synthetic_data(num_samples=num_synthetic)
        
        # Combine datasets
        combined_data = pd.concat([real_data, synthetic_data], ignore_index=True)
        
        return combined_data
