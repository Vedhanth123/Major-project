import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from scipy.stats import vonmises

# Load Dataset (Kaggle Credit Card Fraud Dataset)
df = pd.read_csv("creditcard.csv")

# Normalize 'Amount' Feature
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df.drop(['Time'], axis=1, inplace=True)  # Drop Time column

# Feature Engineering: Transaction Aggregation
def aggregate_transactions(df, hours=24):
    df['Transaction_Hour'] = np.random.randint(0, 24, df.shape[0])  # Simulated Transaction Hours
    df['Rolling_Amount'] = df.groupby('Class')['Amount'].transform(lambda x: x.rolling(hours, min_periods=1).sum())
    df['Rolling_Count'] = df.groupby('Class')['Amount'].transform(lambda x: x.rolling(hours, min_periods=1).count())
    return df

df = aggregate_transactions(df)

# Feature Engineering: Periodic Behavioral Features (Von Mises Distribution)
def compute_periodic_feature(df):
    df['Transaction_Angle'] = df['Transaction_Hour'] * (2 * np.pi / 24)
    mean_angle = np.arctan2(np.sin(df['Transaction_Angle']).mean(), np.cos(df['Transaction_Angle']).mean())
    df['Periodic_Deviation'] = np.abs(df['Transaction_Angle'] - mean_angle)
    return df

df = compute_periodic_feature(df)

# Split Dataset
X = df.drop(columns=['Class'])
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Train Logistic Regression Classifier
lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Evaluation
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_rf))

print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_lr))

# Plot Feature Importance
plt.figure(figsize=(10, 5))
plt.barh(X.columns, rf_model.feature_importances_)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.show()