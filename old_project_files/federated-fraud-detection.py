import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from imblearn.over_sampling import SMOTE

class FederatedFraudDetection:
    def __init__(self, num_banks=100, learning_rate=0.01, epochs=5, batch_size=80):
        self.num_banks = num_banks
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.global_model = None

    def load_and_preprocess_data(self, filepath):
        # Load dataset
        data = pd.read_csv(filepath)
        
        # Separate features and target
        X = data.drop(['Class'], axis=1)
        y = data['Class']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply SMOTE for balancing
        smote = SMOTE(sampling_strategy='auto')
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
        
        return train_test_split(X_resampled, y_resampled, test_size=0.2)

    def create_cnn_model(self, input_shape):
        model = Sequential([
            Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def simulate_bank_training(self, X_train, y_train):
        # Simulate splitting data across banks
        bank_models = []
        bank_updates = []
        
        # Split data into bank-specific subsets
        bank_size = len(X_train) // self.num_banks
        
        for i in range(self.num_banks):
            start = i * bank_size
            end = (i + 1) * bank_size
            
            bank_X = X_train[start:end]
            bank_y = y_train[start:end]
            
            # Create local model based on global model
            local_model = self.create_cnn_model((bank_X.shape[1], 1))
            
            # Train local model
            local_model.fit(
                bank_X.reshape(bank_X.shape[0], bank_X.shape[1], 1), 
                bank_y, 
                epochs=self.epochs, 
                batch_size=self.batch_size, 
                verbose=0
            )
            
            # Store local model and its weights
            bank_models.append(local_model)
            bank_updates.append(local_model.get_weights())
        
        return bank_models, bank_updates

    def federated_averaging(self, bank_updates):
        # Aggregate model updates from different banks
        averaged_weights = []
        num_banks = len(bank_updates)
        
        # Compute average of weights across all banks
        for weights_list in zip(*bank_updates):
            avg_weight = np.mean(weights_list, axis=0)
            averaged_weights.append(avg_weight)
        
        return averaged_weights

    def train_federated_model(self, filepath):
        # Load and preprocess data
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data(filepath)
        
        # Initialize global model
        self.global_model = self.create_cnn_model((X_train.shape[1], 1))
        
        # Simulate federated training
        for round in range(10):  # 10 communication rounds
            # Simulate bank-level training
            bank_models, bank_updates = self.simulate_bank_training(
                X_train, y_train
            )
            
            # Perform federated averaging
            averaged_weights = self.federated_averaging(bank_updates)
            
            # Update global model with averaged weights
            self.global_model.set_weights(averaged_weights)
        
        # Evaluate final model
        y_pred = self.global_model.predict(
            X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        )
        
        return y_pred, y_test

# Example usage
if __name__ == "__main__":
    # Replace with actual path to your downloaded dataset
    filepath = 'creditcard.csv'
    
    ffd = FederatedFraudDetection()
    y_pred, y_test = ffd.train_federated_model(filepath)
    
    # Compute performance metrics
    from sklearn.metrics import roc_auc_score, f1_score
    print("AUC Score:", roc_auc_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, (y_pred > 0.5).astype(int)))
