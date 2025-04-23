import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc

# Load Dataset (Kaggle Credit Card Fraud Dataset)
df = pd.read_csv("creditcard.csv")

# Normalize 'Amount' Feature
df['Amount'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
df.drop(['Time'], axis=1, inplace=True)  # Drop Time column

# Separate Features & Labels
X = df.drop(columns=['Class'])
y = df['Class']

# Handle Class Imbalance using SMOTE
smote = SMOTE(sampling_strategy=0.2, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Train Models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=100, random_state=42)
mlp_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test)
y_pred_mlp = mlp_model.predict(X_test)

# Probability Scores for AUC
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]
y_prob_mlp = mlp_model.predict_proba(X_test)[:, 1]

# Evaluation
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob_rf))

print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob_svm))

print("Neural Network Classification Report:")
print(classification_report(y_test, y_pred_mlp))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob_mlp))

# Precision-Recall Curve
plt.figure(figsize=(10, 6))
for model_name, y_prob in zip(['Random Forest', 'SVM', 'Neural Network'], [y_prob_rf, y_prob_svm, y_prob_mlp]):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{model_name} (AUC={pr_auc:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
