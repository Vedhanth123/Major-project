import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from fcmeans import FCM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load Dataset (Kaggle Credit Card Fraud Dataset)
df = pd.read_csv("creditcard.csv")

# Normalize Features
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df.drop(['Time'], axis=1, inplace=True)  # Drop Time column

# Separate Features & Labels
X = df.drop(columns=['Class'])
y = df['Class']

# Apply Fuzzy C-Means Clustering
fcm = FCM(n_clusters=2)
fcm.fit(X)
cluster_centers = fcm.centers
labels = fcm.predict(X)
df['Cluster'] = labels

# Compute Suspicion Score (SC) using Euclidean Distance
def compute_suspicion_score(X, cluster_centers):
    distances = np.linalg.norm(X - cluster_centers[labels], axis=1)
    return distances

df['Suspicion_Score'] = compute_suspicion_score(X.values, cluster_centers)

# Define Thresholds
Lth = np.percentile(df['Suspicion_Score'], 25)  # Lower threshold
Uth = np.percentile(df['Suspicion_Score'], 75)  # Upper threshold

def classify_transaction(score, Lth, Uth):
    if score < Lth:
        return 'Legitimate'
    elif Lth <= score <= Uth:
        return 'Suspicious'
    else:
        return 'Fraudulent'

df['Transaction_Type'] = df['Suspicion_Score'].apply(lambda x: classify_transaction(x, Lth, Uth))

# Filter Suspicious Transactions for Neural Network Training
suspicious_data = df[df['Transaction_Type'] == 'Suspicious']
X_suspicious = suspicious_data.drop(columns=['Class', 'Transaction_Type'])
y_suspicious = suspicious_data['Class']

# Split Data for NN Training
X_train, X_test, y_train, y_test = train_test_split(X_suspicious, y_suspicious, test_size=0.2, random_state=42)

# Build Neural Network Model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate Model
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.show()
