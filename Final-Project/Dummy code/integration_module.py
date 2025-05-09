import os
import numpy as np
import logging
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('EmotionRecommender')

class NeuralRecommender:
    """Neural network-based video recommendation system"""
    
    def __init__(self, emotion_nn, feature_extractor, 
                 data_generator, youtube_recommender,
                 min_confidence=0.6, model_dir='models'):
        """Initialize the neural recommendation system
        
        Args:
            emotion_nn: EmotionVideoNN instance
            feature_extractor: UserFeatureExtractor instance
            data_generator: TrainingDataGenerator instance
            youtube_recommender: YouTubeRecommender instance
            min_confidence: Minimum confidence threshold for recommendations
            model_dir: Directory to store model and training data
        """
        self.emotion_nn = emotion_nn
        self.feature_extractor = feature_extractor
        self.data_generator = data_generator
        self.youtube = youtube_recommender
        self.min_confidence = min_confidence
        self.model_dir = model_dir
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Load model if it exists
        self.model_loaded = self._load_model()
        
    def _load_model(self):
        """Load the neural network model if it exists
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            return self.emotion_nn.load_model()
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
            return False
        
    def train_model(self, force_retrain=False):
        """Train or update the neural network model
        
        Args:
            force_retrain: Force retraining even if model exists
            
        Returns:
            bool: True if training was successful
        """
        if self.model_loaded and not force_retrain:
            logger.info("Model already loaded, skipping training")
            return True
            
        try:
            logger.info("Generating training data...")
            training_data = self.data_generator.get_combined_dataset()
            
            if len(training_data) < 100:
                logger.warning("Not enough training data (< 100 samples)")
                # Generate more synthetic data if needed
                synthetic_data = self.data_generator.generate_synthetic_data(1000)
                training_data = synthetic_data
            
            logger.info(f"Training model with {len(training_data)} samples...")
            