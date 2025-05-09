import numpy as np
import pandas as pd
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random
import time
from datetime import datetime, timedelta
import os

# Database constants
DATABASE_PATH = "user_data.db"

# Define emotion categories and age ranges (matching your existing system)
EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "neutral", "disgust"]
AGE_RANGES = ["<10", "10-19", "20-29", "30-39", "40-49", "50-59", "60+"]

# YouTube categories (matching your existing system)
YOUTUBE_CATEGORIES = {
    "22": "Music",
    "23": "Comedy",
    "24": "Entertainment",
    "25": "News & Politics",
    "26": "Howto & Style",
    "27": "Education",
    "28": "Science & Technology"
}

class EmotionDataset(Dataset):
    """Dataset for emotion-based recommendation data"""
    
    def __init__(self, features, targets=None, train=True):
        self.features = torch.tensor(features, dtype=torch.float32)
        
        if train:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None
            
        self.train = train
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.train:
            return self.features[idx], self.targets[idx]
        else:
            return self.features[idx]


class EmotionRecommenderNN(nn.Module):
    """Neural network model for emotion-based content recommendations"""
    
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        super(EmotionRecommenderNN, self).__init__()
        
        # Create layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
            
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())  # For multi-label classification
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class RecommenderDataGenerator:
    """Class to generate synthetic training data for the recommendation model"""
    
    def __init__(self):
        self.emotions = EMOTIONS
        self.age_ranges = AGE_RANGES
        self.category_ids = list(YOUTUBE_CATEGORIES.keys())
        
        # Load existing schema for a better understanding of real data
        self._load_database_schema()
    
    def _load_database_schema(self):
        """Examine existing database schema"""
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            # Get emotion_logs table schema
            cursor.execute("PRAGMA table_info(emotion_logs)")
            self.emotion_logs_schema = cursor.fetchall()
            
            # Get recommended_videos table schema if it exists
            try:
                cursor.execute("PRAGMA table_info(recommended_videos)")
                self.video_schema = cursor.fetchall()
            except:
                self.video_schema = []
                
            # Get user_interactions table schema if it exists
            try:
                cursor.execute("PRAGMA table_info(user_interactions)")
                self.interactions_schema = cursor.fetchall()
            except:
                self.interactions_schema = []
                
            conn.close()
            print("Database schema loaded successfully")
            
        except Exception as e:
            print(f"Error loading database schema: {e}")
            # Create default schemas
            self.emotion_logs_schema = [
                (0, 'id', 'INTEGER', 0, None, 1),
                (1, 'name', 'TEXT', 1, None, 0),
                (2, 'age_range', 'TEXT', 1, None, 0),
                (3, 'emotion', 'TEXT', 1, None, 0),
                (4, 'timestamp', 'INTEGER', 1, None, 0),
                (5, 'city', 'TEXT', 0, None, 0),
                (6, 'country', 'TEXT', 0, None, 0)
            ]
            
    def _generate_demographics(self, n_samples):
        """Generate synthetic demographic data"""
        demographics = []
        
        for _ in range(n_samples):
            name = f"User_{random.randint(1, 100)}"
            age_range = random.choice(self.age_ranges)
            city = random.choice(["Hyderabad","Mumbai","Chennai","Kolkata", "Delhi", "Bangalore", "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Visakhapatnam", "Surat", "Vadodara", "Indore", "Coimbatore", "Patna", "Bhopal", "Thane", "Agra", "Nashik", "Faridabad", "Meerut", "Rajkot", "Kalyan-Dombivli", "Vasai-Virar", "Varanasi", "Srinagar", "Aurangabad", "Dhanbad"])
            country = "India"
            
            demographics.append({
                "name": name,
                "age_range": age_range,
                "city": city,
                "country": country
            })
            
        return demographics
    
    def _generate_emotion_logs(self, n_samples):
        """Generate synthetic emotion logs"""
        emotion_logs = []
        demographics = self._generate_demographics(n_samples // 10)  # Create fewer unique users
        
        # Current timestamp
        current_time = int(time.time())
        
        for _ in range(n_samples):
            # Pick a random demographic
            demo = random.choice(demographics)
            
            # Create emotion log
            emotion = random.choice(self.emotions)
            
            # Random timestamp within the last 30 days
            timestamp = current_time - random.randint(0, 30 * 24 * 60 * 60)
            
            emotion_logs.append({
                "name": demo["name"],
                "age_range": demo["age_range"],
                "emotion": emotion,
                "timestamp": timestamp,
                "city": demo["city"],
                "country": demo["country"]
            })
            
        return pd.DataFrame(emotion_logs)
    
    def save_data_to_csv(self, emotion_logs, interactions, emotion_logs_path='data/emotion_logs.csv', interactions_path='data/user_interactions.csv'):
        """Save generated data to CSV files for future use"""
        try:
            # Create directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            # Save to CSV
            emotion_logs.to_csv(emotion_logs_path, index=False)
            interactions.to_csv(interactions_path, index=False)
            
            print(f"Data saved to CSV files: {emotion_logs_path} and {interactions_path}")
            return True
        except Exception as e:
            print(f"Error saving data to CSV: {e}")
            return False
    
    def load_data_from_csv(self, emotion_logs_path='data/emotion_logs.csv', interactions_path='data/user_interactions.csv'):
        """Load data from CSV files instead of database"""
        try:
            if os.path.exists(emotion_logs_path) and os.path.exists(interactions_path):
                print("Loading data from CSV files...")
                emotion_logs = pd.read_csv(emotion_logs_path)
                interactions = pd.read_csv(interactions_path)
                
                # Make sure the timestamp column exists
                if 'timestamp' not in emotion_logs.columns:
                    emotion_logs['timestamp'] = int(time.time())
                
                # Check if necessary columns exist in interactions
                if 'timestamp' not in interactions.columns:
                    interactions['timestamp'] = int(time.time())
                
                return emotion_logs, interactions
            else:
                print("CSV files not found")
                return None, None
        except Exception as e:
            print(f"Error loading data from CSV: {e}")
            return None, None
        
    def _generate_category_preferences(self):
        """Generate synthetic category preferences based on emotions and age ranges"""
        preferences = {}
        
        # For each emotion and age range combination, create category preferences
        for emotion in self.emotions:
            for age_range in self.age_ranges:
                # Initialize with small random values
                prefs = {cat_id: random.uniform(0.1, 0.3) for cat_id in self.category_ids}
                
                # Set stronger preferences based on your domain knowledge
                # These values should align with your existing mapping logic
                if emotion == "happy":
                    prefs["22"] += 0.5  # Music
                    prefs["23"] += 0.5  # Comedy
                    prefs["24"] += 0.3  # Entertainment
                elif emotion == "sad":
                    prefs["22"] += 0.5  # Music
                    prefs["25"] += 0.3  # News & Politics
                    prefs["27"] += 0.4  # Education
                elif emotion == "angry":
                    prefs["22"] += 0.4  # Music
                    prefs["28"] += 0.4  # Science & Technology
                elif emotion == "fear":
                    prefs["22"] += 0.6  # Music
                    prefs["26"] += 0.4  # Howto & Style
                elif emotion == "surprise":
                    prefs["28"] += 0.5  # Science & Technology
                    prefs["24"] += 0.4  # Entertainment
                    prefs["27"] += 0.3  # Education
                elif emotion == "neutral":
                    prefs["27"] += 0.5  # Education
                    prefs["28"] += 0.4  # Science & Technology
                    prefs["26"] += 0.3  # Howto & Style
                elif emotion == "disgust":
                    prefs["24"] += 0.5  # Entertainment
                    prefs["26"] += 0.5  # Howto & Style
                
                # Age-specific preferences
                if age_range == "<10":
                    prefs["24"] += 0.2  # Entertainment
                    prefs["27"] += 0.3  # Education
                elif "10-19" in age_range:
                    prefs["23"] += 0.3  # Comedy
                    prefs["24"] += 0.2  # Entertainment
                elif "20-29" in age_range:
                    prefs["28"] += 0.2  # Science & Technology
                    prefs["22"] += 0.2  # Music
                elif "30-39" in age_range:
                    prefs["25"] += 0.2  # News & Politics
                    prefs["26"] += 0.2  # Howto & Style
                elif "40-49" in age_range:
                    prefs["25"] += 0.3  # News & Politics
                    prefs["27"] += 0.2  # Education
                elif "50-59" in age_range:
                    prefs["25"] += 0.4  # News & Politics
                    prefs["27"] += 0.3  # Education
                else:  # 60+
                    prefs["25"] += 0.4  # News & Politics
                    prefs["26"] += 0.3  # Howto & Style
                
                # Add some noise
                for cat_id in prefs:
                    prefs[cat_id] += random.uniform(-0.05, 0.05)
                    # Ensure values are between 0 and 1
                    prefs[cat_id] = max(0.0, min(1.0, prefs[cat_id]))
                
                preferences[(emotion, age_range)] = prefs
                
        return preferences
    
    def _generate_video_interactions(self, emotion_logs, n_interactions=5000):
        """Generate synthetic video interactions data"""
        interactions = []
        
        # Generate category preferences
        category_preferences = self._generate_category_preferences()
        
        # For each interaction
        for _ in range(n_interactions):
            # Randomly select a row from emotion_logs
            emotion_log = emotion_logs.iloc[random.randint(0, len(emotion_logs) - 1)]
            
            name = emotion_log["name"]
            emotion = emotion_log["emotion"]
            age_range = emotion_log["age_range"]
            timestamp = emotion_log["timestamp"]
            
            # Get preference for this demographic
            prefs = category_preferences[(emotion, age_range)]
            
            # Select a category based on preferences
            categories = list(prefs.keys())
            weights = list(prefs.values())
            category_id = random.choices(categories, weights=weights, k=1)[0]
            
            # Generate a random video ID
            video_id = f"v{random.randint(10000, 99999)}"
            
            # Generate interaction metrics
            # Users are more likely to interact with content matching their preferences
            match_score = prefs[category_id]
            
            watched_duration = random.uniform(0.0, 1.0) * (0.5 + match_score)  # Higher for better matches
            liked = random.random() < match_score  # More likely to like if good match
            
            # Generate random watch time between 0 and 10 minutes
            watch_time_secs = int(watched_duration * 600)  
            
            interactions.append({
                "name": name,
                "video_id": video_id,
                "category_id": category_id,
                "timestamp": timestamp + random.randint(60, 3600),  # Sometime after emotion was detected
                "watched_percentage": watched_duration,
                "liked": liked,
                "watch_time_secs": watch_time_secs
            })
            
        return pd.DataFrame(interactions)
    
    def generate_training_data(self, n_emotion_samples=1000, n_interactions=5000):
        """Generate synthetic training data"""
        print("Generating synthetic training data...")
        
        # Generate emotion logs
        emotion_logs = self._generate_emotion_logs(n_emotion_samples)
        
        # Generate video interactions
        interactions = self._generate_video_interactions(emotion_logs, n_interactions)
        
        return emotion_logs, interactions
    
    def save_to_database(self, emotion_logs, interactions):
        """Save synthetic data to the database"""
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            
            # Create tables if they don't exist
            conn.execute('''
                CREATE TABLE IF NOT EXISTS emotion_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    age_range TEXT NOT NULL,
                    emotion TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    city TEXT,
                    country TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    category_id TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    watched_percentage REAL,
                    liked INTEGER,
                    watch_time_secs INTEGER
                )
            ''')
            
            # Save emotion logs
            emotion_logs.to_sql('emotion_logs', conn, if_exists='append', index=False)
            
            # Save interactions
            interactions.to_sql('user_interactions', conn, if_exists='append', index=False)
            
            conn.commit()
            conn.close()
            
            print(f"Successfully saved {len(emotion_logs)} emotion logs and {len(interactions)} interactions to database")
            
        except Exception as e:
            print(f"Error saving to database: {e}")
    
    def load_from_database(self):
        """Load training data from database"""
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            
            # Load emotion logs
            emotion_logs = pd.read_sql("SELECT * FROM emotion_logs", conn)
            
            # Load interactions if table exists
            try:
                interactions = pd.read_sql("SELECT * FROM user_interactions", conn)
            except:
                interactions = pd.DataFrame()
                
            conn.close()
            
            # Check if necessary columns exist
            required_columns = ['name', 'age_range', 'emotion', 'timestamp']
            for col in required_columns:
                if col not in emotion_logs.columns:
                    print(f"Error: Required column '{col}' not found in emotion_logs table")
                    # Add the missing column with default values
                    if col == 'timestamp':
                        emotion_logs[col] = int(time.time())
                    else:
                        emotion_logs[col] = "unknown"
            
            if not interactions.empty:
                required_int_columns = ['name', 'video_id', 'category_id', 'timestamp', 
                                       'watched_percentage', 'liked', 'watch_time_secs']
                for col in required_int_columns:
                    if col not in interactions.columns:
                        print(f"Error: Required column '{col}' not found in user_interactions table")
                        # Add the missing column with default values
                        if col == 'timestamp':
                            interactions[col] = int(time.time())
                        elif col in ['watched_percentage']:
                            interactions[col] = 0.5
                        elif col in ['liked', 'watch_time_secs']:
                            interactions[col] = 0
                        else:
                            interactions[col] = "unknown"
            
            return emotion_logs, interactions
            
        except Exception as e:
            print(f"Error loading from database: {e}")
            return None, None


class EmotionRecommenderTrainer:
    """Class to train and evaluate the emotion recommender model"""
    
    def __init__(self, input_size, hidden_sizes, output_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = EmotionRecommenderNN(input_size, hidden_sizes, output_size)
        self.model.to(self.device)
        
        # Initialize optimizers and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        
        # For tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # For data preprocessing
        self.feature_scaler = StandardScaler()
        self.label_encoder_emotion = LabelEncoder()
        self.label_encoder_age = LabelEncoder()
        
        # Fit the encoders
        self.label_encoder_emotion.fit(EMOTIONS)
        self.label_encoder_age.fit(AGE_RANGES)
        
        # Create model directory
        os.makedirs('models', exist_ok=True)
    
    def prepare_training_data(self, emotion_logs, interactions):
        """Prepare training data from logs and interactions"""
        if interactions.empty:
            print("No interaction data available. Cannot prepare training data.")
            return None, None
            
        print("Preparing training data...")
        
        # Ensure required columns exist
        required_columns = ['name', 'age_range', 'emotion', 'timestamp']
        for col in required_columns:
            if col not in emotion_logs.columns:
                print(f"Error: Required column '{col}' not found in emotion_logs table")
                return None, None
        
        required_int_columns = ['name', 'video_id', 'category_id', 'timestamp', 
                               'watched_percentage', 'liked', 'watch_time_secs']
        for col in required_int_columns:
            if col not in interactions.columns:
                print(f"Error: Required column '{col}' not found in user_interactions table")
                return None, None
        
        # Merge emotion_logs with interactions
        try:
            merged_data = pd.merge(interactions, emotion_logs, on='name', suffixes=('_int', '_emo'))
        except Exception as e:
            print(f"Error merging data: {e}")
            return None, None
        
        # Fix timestamp columns if there's confusion
        if 'timestamp' not in merged_data.columns:
            if 'timestamp_int' in merged_data.columns:
                merged_data['timestamp'] = merged_data['timestamp_int']
            elif 'timestamp_emo' in merged_data.columns:
                merged_data['timestamp'] = merged_data['timestamp_emo']
        
        # Extract features
        features = []
        targets = []
        
        # Process each row
        for _, row in merged_data.iterrows():
            try:
                # One-hot encode emotion
                emotion_idx = self.label_encoder_emotion.transform([row['emotion']])[0]
                emotion_one_hot = np.zeros(len(EMOTIONS))
                emotion_one_hot[emotion_idx] = 1
                
                # One-hot encode age_range
                age_idx = self.label_encoder_age.transform([row['age_range']])[0]
                age_one_hot = np.zeros(len(AGE_RANGES))
                age_one_hot[age_idx] = 1
                
                # Get timestamp for time features
                try:
                    timestamp_val = int(row['timestamp'])
                    dt = datetime.fromtimestamp(timestamp_val)
                    hour_normalized = dt.hour / 23.0
                    weekday_normalized = dt.weekday() / 6.0
                except (ValueError, TypeError, OverflowError):
                    print(f"Invalid timestamp value: {row['timestamp']}, using current time")
                    dt = datetime.now()
                    hour_normalized = dt.hour / 23.0
                    weekday_normalized = dt.weekday() / 6.0
                
                # Create feature vector
                feature = np.concatenate([
                    emotion_one_hot,
                    age_one_hot,
                    [1.0 if str(row.get('country', '')).strip() == country else 0.0 for country in ['India']],
                    [hour_normalized],
                    [weekday_normalized],
                ])
                
                # Create target (one-hot encoded category_id)
                target = np.zeros(len(YOUTUBE_CATEGORIES))
                category_id = str(row['category_id'])
                if category_id in YOUTUBE_CATEGORIES:
                    target_idx = list(YOUTUBE_CATEGORIES.keys()).index(category_id)
                    target[target_idx] = 1
                    
                    features.append(feature)
                    targets.append(target)
                else:
                    print(f"Warning: Unknown category ID: {category_id}")
                
            except Exception as e:
                print(f"Error processing row: {e}")
                continue
        
        if not features:
            print("No valid features could be extracted")
            return None, None
            
        # Convert to numpy arrays
        features = np.array(features)
        targets = np.array(targets)
        
        # Normalize features
        features = self.feature_scaler.fit_transform(features)
        
        print(f"Prepared {len(features)} training samples with {features.shape[1]} features")
        return features, targets
    
    def train(self, features, targets, epochs=50, batch_size=64, validation_split=0.2):
        """Train the model"""
        if features is None or targets is None:
            print("No training data available")
            return
            
        print(f"Training model with {len(features)} samples...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, targets, test_size=validation_split, random_state=42
        )
        
        # Create datasets
        train_dataset = EmotionDataset(X_train, y_train)
        val_dataset = EmotionDataset(X_val, y_val)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Train the model
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_features, batch_targets in train_loader:
                # Move data to device
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    # Move data to device
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(batch_features)
                    loss = self.criterion(outputs, batch_targets)
                    
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(val_loader)
            self.val_losses.append(avg_val_loss)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                
            # Save best model
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save_model('models/best_emotion_recommender.pt')
                
        # Save final model
        self.save_model('models/final_emotion_recommender.pt')
        
        # Plot training history
        self.plot_training_history()
        
        return self.train_losses, self.val_losses
    
    def save_model(self, path):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'feature_scaler': self.feature_scaler,
            'label_encoder_emotion': self.label_encoder_emotion,
            'label_encoder_age': self.label_encoder_age,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model from disk"""
        if not os.path.exists(path):
            print(f"Model file {path} not found")
            return False
            
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Properly load and restore the scalers and encoders
            self.feature_scaler = checkpoint['feature_scaler']
            self.label_encoder_emotion = checkpoint['label_encoder_emotion']
            self.label_encoder_age = checkpoint['label_encoder_age']
            
            # Load the rest of the attributes
            self.best_val_loss = checkpoint['best_val_loss']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def plot_training_history(self):
        """Plot training and validation loss history"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Training Loss')
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('models/training_history.png')
        plt.close()
        
    def evaluate(self, features, targets):
        """Evaluate model on test data"""
        if features is None or targets is None:
            print("No evaluation data available")
            return
        
        # Create dataset and loader
        test_dataset = EmotionDataset(features, targets)
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        # Evaluate model
        self.model.eval()
        test_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                # Move data to device
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_targets)
                
                test_loss += loss.item()
                
                # Store predictions and targets
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(batch_targets.cpu().numpy())
                
        # Calculate metrics
        avg_test_loss = test_loss / len(test_loader)
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Calculate accuracy (for top prediction)
        pred_indices = np.argmax(all_predictions, axis=1)
        target_indices = np.argmax(all_targets, axis=1)
        accuracy = np.mean(pred_indices == target_indices)
        
        # Calculate top-3 accuracy
        top3_accuracy = 0
        for i in range(len(all_predictions)):
            pred_idx = np.argsort(all_predictions[i])[-3:]
            target_idx = np.argmax(all_targets[i])
            if target_idx in pred_idx:
                top3_accuracy += 1
        top3_accuracy /= len(all_predictions)
        
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Top-3 Accuracy: {top3_accuracy:.4f}")
        
        # Plot confusion matrix
        self._plot_confusion_matrix(target_indices, pred_indices)
        
        return avg_test_loss, accuracy, top3_accuracy
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        category_names = list(YOUTUBE_CATEGORIES.values())
        
        plt.figure(figsize=(12, 10))
        cm = np.zeros((len(category_names), len(category_names)))
        
        for i in range(len(y_true)):
            cm[y_true[i]][y_pred[i]] += 1
            
        # Normalize by row
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = cm / row_sums
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=category_names, yticklabels=category_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png')
        plt.close()
    
        
    def predict(self, user_name, emotion, age_range, timestamp=None, country=None):
        """Predict category preferences for a user"""
        if timestamp is None:
            timestamp = int(time.time())
            
        if country is None:
            country = "India"  # Changed default to India to match the training data
            
        # Prepare feature vector
        emotion_idx = self.label_encoder_emotion.transform([emotion])[0]
        emotion_one_hot = np.zeros(len(EMOTIONS))
        emotion_one_hot[emotion_idx] = 1
        
        age_idx = self.label_encoder_age.transform([age_range])[0]
        age_one_hot = np.zeros(len(AGE_RANGES))
        age_one_hot[age_idx] = 1
        
        # Country feature - matches what's used in prepare_training_data
        # Only use 'India' to match the feature dimension
        country_feature = [1.0 if country == 'India' else 0.0]
        
        # Time features
        dt = datetime.fromtimestamp(timestamp)
        hour_normalized = dt.hour / 23.0
        weekday_normalized = dt.weekday() / 6.0
        
        # Create feature vector - matching the structure in prepare_training_data
        feature = np.concatenate([
            emotion_one_hot,
            age_one_hot,
            country_feature,
            [hour_normalized],
            [weekday_normalized]
        ])
        
        # Ensure scaler is fitted before using it
        if not hasattr(self.feature_scaler, 'mean_') or self.feature_scaler.mean_ is None:
            print("Warning: Feature scaler not fitted. Using raw features.")
            feature_normalized = feature.reshape(1, -1)
        else:
            # Normalize feature
            feature_normalized = self.feature_scaler.transform(feature.reshape(1, -1))
        
        # Convert to tensor
        feature_tensor = torch.tensor(feature_normalized, dtype=torch.float32).to(self.device)
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(feature_tensor).cpu().numpy()[0]
            
        # Convert to category preferences
        category_prefs = {}
        for i, cat_id in enumerate(YOUTUBE_CATEGORIES.keys()):
            category_prefs[cat_id] = float(prediction[i])
            
        return category_prefs

class ContinuousLearningSystem:
    """Class to implement continuous learning from user interactions"""
    
    def __init__(self, recommender_trainer, update_interval=24*60*60):  # Default: update daily
        self.trainer = recommender_trainer
        self.update_interval = update_interval
        self.last_update_time = int(time.time())
        
        # Load latest model if available
        self.trainer.load_model('models/best_emotion_recommender.pt')
        
    def update_model(self, force=False):
        """Check if it's time to update the model and do so if needed"""
        current_time = int(time.time())
        
        # Check if update interval has passed or force update
        if force or (current_time - self.last_update_time) >= self.update_interval:
            print("Updating model with new interaction data...")
            
            # Load all data from database or CSV
            data_generator = RecommenderDataGenerator()
            
            # Try loading from CSV first
            csv_emotion_logs_path = 'data/emotion_logs.csv'
            csv_interactions_path = 'data/user_interactions.csv'
            
            if os.path.exists(csv_emotion_logs_path) and os.path.exists(csv_interactions_path):
                emotion_logs, interactions = data_generator.load_data_from_csv()
            else:
                # Fallback to database
                emotion_logs, interactions = data_generator.load_from_database()
            
            if emotion_logs is not None and not interactions.empty:
                # Prepare training data
                features, targets = self.trainer.prepare_training_data(emotion_logs, interactions)
                
                if features is not None and targets is not None:
                    # Train model with new data
                    self.trainer.train(features, targets, epochs=20, batch_size=64)
                    
                    # Update last update time
                    self.last_update_time = current_time
                    
                    print("Model updated successfully")
                    return True
            
            print("No new data available for model update")
            return False

    def log_interaction(self, name, video_id, category_id, watched_percentage, liked, watch_time_secs):
        """Log a user interaction to the database"""
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            # Insert interaction
            cursor.execute('''
                INSERT INTO user_interactions (name, video_id, category_id, timestamp, watched_percentage, liked, watch_time_secs)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (name, video_id, category_id, int(time.time()), watched_percentage, 1 if liked else 0, watch_time_secs))
            
            conn.commit()
            conn.close()
            
            print(f"Interaction logged: {name}, {video_id}, {category_id}")
            return True
            
        except Exception as e:
            print(f"Error logging interaction: {e}")
            return False
    
    def get_recommendation(self, name, emotion, age_range, country=None, top_n=3):
        """Get recommendations for a user based on their current emotion"""
        # Get category preferences
        category_prefs = self.trainer.predict(name, emotion, age_range, country=country)
        
        # Sort categories by preference score
        sorted_categories = sorted(category_prefs.items(), key=lambda x: x[1], reverse=True)
        
        # Get top N categories
        top_categories = sorted_categories[:top_n]
        
        # Convert to readable format
        recommendations = []
        for cat_id, score in top_categories:
            recommendations.append({
                "category": YOUTUBE_CATEGORIES[cat_id],
                "category_id": cat_id,
                "score": score
            })
            
        return recommendations

def main():
    """Main function to demonstrate system functionality"""
    print("Emotion-Based Content Recommendation System")
    print("==========================================")
    
    # Create data generator
    data_generator = RecommenderDataGenerator()
    
    # Check if CSV files exist
    csv_emotion_logs_path = 'data/emotion_logs.csv'
    csv_interactions_path = 'data/user_interactions.csv'
    
    if os.path.exists(csv_emotion_logs_path) and os.path.exists(csv_interactions_path):
        # Load data from CSV files
        print("Loading data from CSV files...")
        emotion_logs, interactions = data_generator.load_data_from_csv()
    else:
        # Check if database exists
        if not os.path.exists(DATABASE_PATH):
            print("Database not found. Generating synthetic data...")
            emotion_logs, interactions = data_generator.generate_training_data(
                n_emotion_samples=2000,
                n_interactions=10000
            )
            # Save to database
            data_generator.save_to_database(emotion_logs, interactions)
        else:
            print("Database found. Loading existing data...")
            emotion_logs, interactions = data_generator.load_from_database()
            
            # If not enough data, generate more
            if emotion_logs is None or len(emotion_logs) < 100 or interactions.empty:
                print("Not enough data found. Generating synthetic data...")
                emotion_logs, interactions = data_generator.generate_training_data(
                    n_emotion_samples=2000,
                    n_interactions=10000
                )
                data_generator.save_to_database(emotion_logs, interactions)
        
        # Save to CSV for future use
        data_generator.save_data_to_csv(emotion_logs, interactions)
    
    # Initialize a temporary trainer with a placeholder input size
    temp_trainer = EmotionRecommenderTrainer(
        input_size=10,  # Just a placeholder
        hidden_sizes=[64, 32],
        output_size=len(YOUTUBE_CATEGORIES)
    )
    
    # Prepare training data to get actual feature dimensions
    features, targets = temp_trainer.prepare_training_data(emotion_logs, interactions)
    
    if features is None or targets is None:
        print("Error: Could not prepare training data")
        return
        
    input_size = features.shape[1]  # Get actual input size from data
    
    # Re-initialize trainer with the correct input size
    trainer = EmotionRecommenderTrainer(
        input_size=input_size,
        hidden_sizes=[64, 32],
        output_size=len(YOUTUBE_CATEGORIES)
    )
    
    # Check if model exists
    model_path = 'models/best_emotion_recommender.pt'
    if not os.path.exists(model_path):
        print("Model not found. Training new model...")
        
        # Train model
        trainer.train(features, targets, epochs=50, batch_size=64)
        
        # Evaluate model
        trainer.evaluate(features, targets)
    else:
        print("Model found. Loading existing model...")
        if not trainer.load_model(model_path):
            print("Error loading model. Training new model...")
            trainer.train(features, targets, epochs=50, batch_size=64)
            trainer.evaluate(features, targets)
    
    # Initialize continuous learning system
    cl_system = ContinuousLearningSystem(trainer)
    
    # Make sure the feature_scaler is fitted
    if not hasattr(trainer.feature_scaler, 'mean_') or trainer.feature_scaler.mean_ is None:
        print("Feature scaler not fitted. Fitting now...")
        trainer.feature_scaler.fit(features)
    
    # Demo - predict recommendations for different emotions
    user_name = "Demo_User"
    age_range = "30-39"
    country = "India"  # Changed to match training data
    
    print("\nRecommendations for different emotions:")
    print("-------------------------------------")
    
    for emotion in EMOTIONS:
        recommendations = cl_system.get_recommendation(user_name, emotion, age_range, country)
        
        print(f"\nFor emotion '{emotion}':")
        for i, rec in enumerate(recommendations):
            print(f"  {i+1}. {rec['category']} (Score: {rec['score']:.3f})")
    
    # Demo - log a new interaction
    print("\nLogging a new interaction...")
    cl_system.log_interaction(
        name=user_name,
        video_id="v12345",
        category_id="23",  # Comedy
        watched_percentage=0.85,
        liked=True,
        watch_time_secs=300
    )
    
    # Demo - update model with new data
    print("\nForcing model update with new interaction data...")
    cl_system.update_model(force=True)
    
    print("\nSystem demonstration complete.")


class EmotionRecommenderAPI:
    """Class to provide API endpoints for the recommendation system"""
    
    def _init_(self):
        # Initialize data generator
        self.data_generator = RecommenderDataGenerator()
        
        # Load emotion logs and interactions
        emotion_logs, interactions = self.data_generator.load_from_database()
        
        # If not enough data, generate more
        if emotion_logs is None or len(emotion_logs) < 100 or interactions.empty:
            emotion_logs, interactions = self.data_generator.generate_training_data(
                n_emotion_samples=2000,
                n_interactions=10000
            )
            self.data_generator.save_to_database(emotion_logs, interactions)
        
        # Initialize a temporary trainer to prepare data and get feature size
        temp_trainer = EmotionRecommenderTrainer(
            input_size=10,  # Just a placeholder
            hidden_sizes=[64, 32],
            output_size=len(YOUTUBE_CATEGORIES)
        )
        
        # Prepare training data to get actual feature dimensions
        features, targets = temp_trainer.prepare_training_data(emotion_logs, interactions)
        input_size = features.shape[1]  # Get actual input size from data
        
        # Initialize trainer with correct input size
        self.trainer = EmotionRecommenderTrainer(
            input_size=input_size,
            hidden_sizes=[64, 32],
            output_size=len(YOUTUBE_CATEGORIES)
        )
        
        # Try to load model
        model_loaded = self.trainer.load_model('models/best_emotion_recommender.pt')
        
        # If model not found, create and train it
        if not model_loaded:
            print("Model not found. Training new model...")
            
            # Prepare training data again with the correct trainer
            features, targets = self.trainer.prepare_training_data(emotion_logs, interactions)
            
            # Train model
            self.trainer.train(features, targets, epochs=50, batch_size=64)
        
        # Initialize continuous learning system
        self.cl_system = ContinuousLearningSystem(self.trainer)
        
    # Rest of the methods remain the same...


class EmotionVisualizationSystem:
    """Class to visualize emotion data and recommendations"""
    
    def __init__(self, database_path=DATABASE_PATH):
        self.database_path = database_path
    
    def load_emotion_data(self):
        """Load emotion data from database"""
        try:
            conn = sqlite3.connect(self.database_path)
            
            # Load emotion logs
            emotion_data = pd.read_sql("SELECT * FROM emotion_logs", conn)
            
            conn.close()
            
            return emotion_data
            
        except Exception as e:
            print(f"Error loading emotion data: {e}")
            return None
    
    def load_interaction_data(self):
        """Load interaction data from database"""
        try:
            conn = sqlite3.connect(self.database_path)
            
            # Load interactions if table exists
            interactions = pd.read_sql("SELECT * FROM user_interactions", conn)
            
            conn.close()
            
            return interactions
            
        except Exception as e:
            print(f"Error loading interaction data: {e}")
            return None
    
    def visualize_emotion_distribution(self):
        """Visualize emotion distribution"""
        emotion_data = self.load_emotion_data()
        
        if emotion_data is None or emotion_data.empty:
            print("No emotion data available")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Count emotions
        emotion_counts = emotion_data['emotion'].value_counts()
        
        # Plot bar chart
        sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
        plt.title('Emotion Distribution')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/emotion_distribution.png')
        plt.close()
        
        print("Emotion distribution visualization saved to 'visualizations/emotion_distribution.png'")
    
    def visualize_emotion_by_age(self):
        """Visualize emotion distribution by age range"""
        emotion_data = self.load_emotion_data()
        
        if emotion_data is None or emotion_data.empty:
            print("No emotion data available")
            return
        
        plt.figure(figsize=(14, 8))
        
        # Create cross-tabulation
        emotion_age_table = pd.crosstab(emotion_data['age_range'], emotion_data['emotion'])
        
        # Plot stacked bar chart
        emotion_age_table.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Emotion Distribution by Age Range')
        plt.xlabel('Age Range')
        plt.ylabel('Count')
        plt.legend(title='Emotion')
        plt.tight_layout()
        plt.savefig('visualizations/emotion_by_age.png')
        plt.close()
        
        print("Emotion by age visualization saved to 'visualizations/emotion_by_age.png'")
    
    def visualize_content_preferences(self):
        """Visualize content preferences by emotion"""
        interaction_data = self.load_interaction_data()
        emotion_data = self.load_emotion_data()
        
        if interaction_data is None or emotion_data is None or interaction_data.empty or emotion_data.empty:
            print("No data available")
            return
        
        # Merge data
        merged_data = pd.merge(interaction_data, emotion_data, on='name')
        
        # Create cross-tabulation
        content_emotion_table = pd.crosstab(merged_data['emotion'], merged_data['category_id'])
        
        # Replace category IDs with names
        content_emotion_table.columns = [YOUTUBE_CATEGORIES[col] for col in content_emotion_table.columns]
        
        # Normalize by row (emotions)
        content_emotion_norm = content_emotion_table.div(content_emotion_table.sum(axis=1), axis=0)
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(content_emotion_norm, annot=True, fmt='.2f', cmap='YlGnBu')
        plt.title('Content Preferences by Emotion')
        plt.ylabel('Emotion')
        plt.xlabel('Content Category')
        plt.tight_layout()
        plt.savefig('visualizations/content_preferences.png')
        plt.close()
        
        print("Content preferences visualization saved to 'visualizations/content_preferences.png'")
    
    def visualize_engagement_metrics(self):
        """Visualize engagement metrics by emotion"""
        interaction_data = self.load_interaction_data()
        emotion_data = self.load_emotion_data()
        
        if interaction_data is None or emotion_data is None or interaction_data.empty or emotion_data.empty:
            print("No data available")
            return
        
        # Merge data
        merged_data = pd.merge(interaction_data, emotion_data, on='name')
        
        # Calculate average watch percentage by emotion
        watch_by_emotion = merged_data.groupby('emotion')['watched_percentage'].mean().reset_index()
        
        # Calculate average watch time by emotion
        time_by_emotion = merged_data.groupby('emotion')['watch_time_secs'].mean().reset_index()
        
        # Calculate like rate by emotion
        merged_data['liked_bool'] = merged_data['liked'] > 0
        like_by_emotion = merged_data.groupby('emotion')['liked_bool'].mean().reset_index()
        
        # Create visualization
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # Watch percentage
        sns.barplot(x='emotion', y='watched_percentage', data=watch_by_emotion, ax=axes[0])
        axes[0].set_title('Average Watch Percentage by Emotion')
        axes[0].set_ylim(0, 1)
        
        # Watch time
        sns.barplot(x='emotion', y='watch_time_secs', data=time_by_emotion, ax=axes[1])
        axes[1].set_title('Average Watch Time (seconds) by Emotion')
        
        # Like rate
        sns.barplot(x='emotion', y='liked_bool', data=like_by_emotion, ax=axes[2])
        axes[2].set_title('Like Rate by Emotion')
        axes[2].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('visualizations/engagement_metrics.png')
        plt.close()
        
        print("Engagement metrics visualization saved to 'visualizations/engagement_metrics.png'")
    
    def visualize_time_patterns(self):
        """Visualize emotion patterns over time"""
        emotion_data = self.load_emotion_data()
        
        if emotion_data is None or emotion_data.empty:
            print("No emotion data available")
            return
        
        # Convert timestamp to datetime
        emotion_data['datetime'] = pd.to_datetime(emotion_data['timestamp'], unit='s')
        
        # Extract hour of day
        emotion_data['hour'] = emotion_data['datetime'].dt.hour
        
        # Extract day of week
        emotion_data['day_of_week'] = emotion_data['datetime'].dt.dayofweek
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Create figures
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        
        # Emotion by hour of day
        hour_emotion = pd.crosstab(emotion_data['hour'], emotion_data['emotion'])
        hour_emotion.plot(kind='line', ax=axes[0])
        axes[0].set_title('Emotions by Hour of Day')
        axes[0].set_xlabel('Hour')
        axes[0].set_ylabel('Count')
        axes[0].set_xticks(range(0, 24))
        axes[0].legend(title='Emotion')
        
        # Emotion by day of week
        day_emotion = pd.crosstab(emotion_data['day_of_week'], emotion_data['emotion'])
        day_emotion.index = day_names
        day_emotion.plot(kind='bar', stacked=True, ax=axes[1])
        axes[1].set_title('Emotions by Day of Week')
        axes[1].set_xlabel('Day of Week')
        axes[1].set_ylabel('Count')
        axes[1].legend(title='Emotion')
        
        plt.tight_layout()
        plt.savefig('visualizations/time_patterns.png')
        plt.close()
        
        print("Time patterns visualization saved to 'visualizations/time_patterns.png'")
    
    def create_all_visualizations(self):
        """Create all visualizations"""
        print("Creating visualizations...")
        
        # Create directory if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)
        
        self.visualize_emotion_distribution()
        self.visualize_emotion_by_age()
        self.visualize_content_preferences()
        self.visualize_engagement_metrics()
        self.visualize_time_patterns()
        
        print("All visualizations created successfully")


if __name__ == "__main__":
    main()               

                           