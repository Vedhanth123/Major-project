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
            
            # Split data into features and labels
            X, y = self.data_generator.split_features_labels(training_data)
            
            # Train the emotion neural network
            training_result = self.emotion_nn.train(X, y)
            
            # Save the model
            save_path = os.path.join(self.model_dir, f"emotion_nn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.model")
            self.emotion_nn.save_model(save_path)
            
            # Log training metrics
            metrics = training_result.get('metrics', {})
            logger.info(f"Training complete - Accuracy: {metrics.get('accuracy', 'N/A')}, Loss: {metrics.get('loss', 'N/A')}")
            
            # Save training metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'samples_count': len(training_data),
                'metrics': metrics,
                'model_path': save_path
            }
            
            with open(os.path.join(self.model_dir, 'training_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
                
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return False
    
    def get_recommendations(self, user_id, emotion_state=None, count=5):
        """Get video recommendations based on user's emotion state
        
        Args:
            user_id: Unique user identifier
            emotion_state: User's current emotion state, or None to detect
            count: Number of recommendations to return
            
        Returns:
            list: List of recommended video dictionaries
        """
        if not self.model_loaded:
            logger.warning("Model not loaded, attempting to train...")
            if not self.train_model():
                logger.error("Could not train model, using fallback recommendations")
                return self.youtube.get_trending_videos(count)
        
        try:
            # Extract user features
            user_features = self.feature_extractor.get_user_features(user_id)
            
            # Detect emotion if not provided
            if emotion_state is None:
                emotion_state = self.feature_extractor.detect_emotional_state(user_id)
                logger.info(f"Detected emotional state for user {user_id}: {emotion_state}")
            
            # Combine user features with emotion state
            input_features = self.feature_extractor.combine_features(user_features, emotion_state)
            
            # Get predictions from neural network
            predictions = self.emotion_nn.predict(input_features)
            
            # Filter predictions based on confidence threshold
            confident_predictions = [p for p in predictions if p.get('confidence', 0) >= self.min_confidence]
            
            if not confident_predictions:
                logger.info("No confident predictions, using fallback recommendations")
                return self.youtube.get_personalized_recommendations(user_id, count)
            
            # Convert predictions to video recommendations
            video_recommendations = self._fetch_videos_for_predictions(confident_predictions, count)
            
            # Log recommendation data
            self._log_recommendation_data(user_id, emotion_state, video_recommendations)
            
            return video_recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return self.youtube.get_trending_videos(count)
    
    def _fetch_videos_for_predictions(self, predictions, count):
        """Fetch videos based on neural network predictions
        
        Args:
            predictions: List of prediction dictionaries
            count: Maximum number of videos to return
            
        Returns:
            list: List of video dictionaries
        """
        videos = []
        
        # Sort predictions by confidence
        sorted_predictions = sorted(predictions, key=lambda x: x.get('confidence', 0), reverse=True)
        
        for prediction in sorted_predictions[:count*2]:  # Get more than needed in case some fail
            try:
                category = prediction.get('category')
                mood = prediction.get('mood')
                
                # Get videos matching the predicted category and mood
                matching_videos = self.youtube.search_videos(
                    category=category,
                    mood=mood,
                    max_results=3
                )
                
                if matching_videos:
                    # Add confidence score to videos
                    for video in matching_videos:
                        video['recommendation_confidence'] = prediction.get('confidence', 0)
                        video['recommendation_reason'] = f"Matches your {mood} mood in {category}"
                    
                    videos.extend(matching_videos)
                    
                    if len(videos) >= count:
                        break
            except Exception as e:
                logger.warning(f"Error fetching videos for prediction {prediction}: {e}")
        
        # Deduplicate and return top videos
        unique_videos = self._deduplicate_videos(videos)
        return unique_videos[:count]
    
    def _deduplicate_videos(self, videos):
        """Remove duplicate videos from a list
        
        Args:
            videos: List of video dictionaries
            
        Returns:
            list: Deduplicated list of videos
        """
        seen_ids = set()
        unique_videos = []
        
        for video in videos:
            video_id = video.get('id')
            if video_id and video_id not in seen_ids:
                seen_ids.add(video_id)
                unique_videos.append(video)
        
        return unique_videos
    
    def _log_recommendation_data(self, user_id, emotion_state, recommendations):
        """Log recommendation data for analysis
        
        Args:
            user_id: User identifier
            emotion_state: User's emotional state
            recommendations: List of recommended videos
        """
        try:
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'emotion_state': emotion_state,
                'recommendations': [{'id': v.get('id'), 'title': v.get('title'), 
                                    'confidence': v.get('recommendation_confidence')} 
                                   for v in recommendations]
            }
            
            log_path = os.path.join(self.model_dir, 'recommendation_logs.jsonl')
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_data) + '\n')
                
        except Exception as e:
            logger.warning(f"Error logging recommendation data: {e}")
    
    def evaluate_model(self, test_data=None):
        """Evaluate the model on test data
        
        Args:
            test_data: Optional test dataset, generated if None
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.model_loaded:
            logger.warning("Model not loaded, cannot evaluate")
            return {'error': 'Model not loaded'}
        
        try:
            # Generate test data if not provided
            if test_data is None:
                logger.info("Generating test data for evaluation...")
                test_data = self.data_generator.generate_test_dataset(500)
            
            # Split data into features and labels
            X_test, y_test = self.data_generator.split_features_labels(test_data)
            
            # Evaluate the model
            metrics = self.emotion_nn.evaluate(X_test, y_test)
            
            logger.info(f"Model evaluation complete - " + 
                       f"Accuracy: {metrics.get('accuracy', 'N/A')}, " +
                       f"Precision: {metrics.get('precision', 'N/A')}, " +
                       f"Recall: {metrics.get('recall', 'N/A')}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {'error': str(e)}
    
    def update_model(self, feedback_data):
        """Update the model based on user feedback
        
        Args:
            feedback_data: List of feedback data points
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Process feedback data
            processed_feedback = self.data_generator.process_feedback_data(feedback_data)
            
            if not processed_feedback:
                logger.warning("No valid feedback data to process")
                return False
            
            logger.info(f"Updating model with {len(processed_feedback)} feedback samples")
            
            # Incrementally update the model
            update_result = self.emotion_nn.update(processed_feedback)
            
            # Save the updated model
            save_path = os.path.join(self.model_dir, f"emotion_nn_updated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.model")
            self.emotion_nn.save_model(save_path)
            
            # Log update metrics
            metrics = update_result.get('metrics', {})
            logger.info(f"Model update complete - Delta Loss: {metrics.get('delta_loss', 'N/A')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            return False
