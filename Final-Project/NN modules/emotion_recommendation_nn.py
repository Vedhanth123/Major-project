import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class EmotionVideoNN:
    """Neural Network for mapping emotions to video preferences"""
    
    def __init__(self, model_path='models/emotion_video_nn.h5', 
                 encoders_path='models/emotion_video_encoders.pkl'):
        """Initialize the neural network for emotion-video mapping
        
        Args:
            model_path: Path to save/load the model
            encoders_path: Path to save/load the encoders
        """
        self.model_path = model_path
        self.encoders_path = encoders_path
        self.model = None
        self.emotion_encoder = None
        self.category_encoder = None
        self.keyword_encoder = None
        self.feature_scaler = None
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
    def build_model(self, emotion_size, user_feature_size=10, 
                    category_size=30, keyword_size=50):
        """Build the neural network architecture
        
        Args:
            emotion_size: Size of emotion one-hot encoding
            user_feature_size: Size of user features (age, watch history, etc.)
            category_size: Size of video category one-hot encoding
            keyword_size: Size of keyword one-hot encoding
        """
        # Input layers
        emotion_input = layers.Input(shape=(emotion_size,), name='emotion_input')
        user_input = layers.Input(shape=(user_feature_size,), name='user_input')
        
        # Emotion path
        emotion_dense = layers.Dense(32, activation='relu')(emotion_input)
        emotion_dropout = layers.Dropout(0.3)(emotion_dense)
        
        # User features path
        user_dense = layers.Dense(24, activation='relu')(user_input)
        user_dropout = layers.Dropout(0.2)(user_dense)
        
        # Merge paths
        merged = layers.Concatenate()([emotion_dropout, user_dropout])
        x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(merged)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        
        # Output layers
        category_output = layers.Dense(category_size, activation='softmax', name='category_output')(x)
        keyword_output = layers.Dense(keyword_size, activation='sigmoid', name='keyword_output')(x)
        
        # Create model
        self.model = models.Model(
            inputs=[emotion_input, user_input],
            outputs=[category_output, keyword_output]
        )
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss={
                'category_output': 'categorical_crossentropy',
                'keyword_output': 'binary_crossentropy'
            },
            metrics={
                'category_output': 'accuracy',
                'keyword_output': 'accuracy'
            }
        )
        
        return self.model
    
    def prepare_data(self, data):
        """Prepare data for training or prediction
        
        Args:
            data: DataFrame with columns: user_id, emotion, features, 
                  watch_history, preferred_categories, preferred_keywords
        
        Returns:
            X_emotions: One-hot encoded emotions
            X_features: User features (scaled)
            y_categories: One-hot encoded preferred categories
            y_keywords: One-hot encoded preferred keywords
        """
        # Initialize encoders if not already done
        if self.emotion_encoder is None:
            self.emotion_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            self.emotion_encoder.fit(data[['emotion']])
            
        if self.category_encoder is None:
            # Extract all unique categories
            all_categories = []
            for categories in data['preferred_categories']:
                all_categories.extend(categories)
            unique_categories = list(set(all_categories))
            self.category_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            self.category_encoder.fit(np.array(unique_categories).reshape(-1, 1))
            
        if self.keyword_encoder is None:
            # Extract all unique keywords
            all_keywords = []
            for keywords in data['preferred_keywords']:
                all_keywords.extend(keywords)
            unique_keywords = list(set(all_keywords))
            self.keyword_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            self.keyword_encoder.fit(np.array(unique_keywords).reshape(-1, 1))
        
        if self.feature_scaler is None:
            self.feature_scaler = StandardScaler()
            self.feature_scaler.fit(np.vstack(data['features'].values))
            
        # Encode emotions
        X_emotions = self.emotion_encoder.transform(data[['emotion']])
        
        # Scale features
        X_features = np.vstack(data['features'].values)
        X_features = self.feature_scaler.transform(X_features)
        
        # Process outputs (categories and keywords)
        y_categories = np.zeros((len(data), len(self.category_encoder.categories_[0])))
        y_keywords = np.zeros((len(data), len(self.keyword_encoder.categories_[0])))
        
        for i, (categories, keywords) in enumerate(zip(data['preferred_categories'], data['preferred_keywords'])):
            if categories:
                cat_encoded = self.category_encoder.transform(np.array(categories).reshape(-1, 1))
                y_categories[i] = cat_encoded.max(axis=0)  # Max to handle multiple categories
                
            if keywords:
                kw_encoded = self.keyword_encoder.transform(np.array(keywords).reshape(-1, 1))
                y_keywords[i] = kw_encoded.max(axis=0)  # Max to handle multiple keywords
        
        return X_emotions, X_features, y_categories, y_keywords
    
    def train(self, data, epochs=50, batch_size=32, validation_split=0.2):
        """Train the model on emotion and user data
        
        Args:
            data: DataFrame with training data
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation data proportion
        
        Returns:
            History object
        """
        X_emotions, X_features, y_categories, y_keywords = self.prepare_data(data)
        
        # Build model if not already built
        if self.model is None:
            self.build_model(
                emotion_size=X_emotions.shape[1],
                user_feature_size=X_features.shape[1],
                category_size=y_categories.shape[1],
                keyword_size=y_keywords.shape[1]
            )
        
        # Train model
        history = self.model.fit(
            [X_emotions, X_features],
            [y_categories, y_keywords],
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5
                )
            ]
        )
        
        # Save model and encoders
        self.save_model()
        
        return history
    
    def predict(self, emotion, user_features):
        """Predict video preferences based on emotion and user features
        
        Args:
            emotion: String emotion state
            user_features: Array of user features
            
        Returns:
            predicted_categories: Top video categories (list)
            predicted_keywords: Top keywords (list)
            category_scores: Category probability scores
            keyword_scores: Keyword probability scores
        """
        if self.model is None:
            self.load_model()
            
        # Encode emotion
        emotion_encoded = self.emotion_encoder.transform(np.array([emotion]).reshape(-1, 1))
        
        # Scale features
        features_scaled = self.feature_scaler.transform(user_features.reshape(1, -1))
        
        # Make prediction
        category_probs, keyword_probs = self.model.predict([emotion_encoded, features_scaled])
        
        # Get top categories and keywords
        top_category_indices = np.argsort(category_probs[0])[::-1][:3]  # Top 3 categories
        top_keyword_indices = np.where(keyword_probs[0] > 0.5)[0]  # Keywords with prob > 0.5
        
        # Decode categories and keywords
        category_names = self.category_encoder.categories_[0]
        keyword_names = self.keyword_encoder.categories_[0]
        
        predicted_categories = [category_names[i] for i in top_category_indices]
        predicted_keywords = [keyword_names[i] for i in top_keyword_indices]
        
        return predicted_categories, predicted_keywords, category_probs, keyword_probs
    
    def save_model(self):
        """Save model and encoders"""
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))
            
        if self.model is not None:
            self.model.save(self.model_path)
            
        # Save encoders and scaler
        joblib.dump({
            'emotion_encoder': self.emotion_encoder,
            'category_encoder': self.category_encoder,
            'keyword_encoder': self.keyword_encoder,
            'feature_scaler': self.feature_scaler
        }, self.encoders_path)
        
    def load_model(self):
        """Load saved model and encoders"""
        if os.path.exists(self.model_path):
            self.model = models.load_model(self.model_path)
            
        if os.path.exists(self.encoders_path):
            encoders = joblib.load(self.encoders_path)
            self.emotion_encoder = encoders['emotion_encoder']
            self.category_encoder = encoders['category_encoder']
            self.keyword_encoder = encoders['keyword_encoder']
            self.feature_scaler = encoders['feature_scaler']
            
        return self.model is not None and self.emotion_encoder is not None