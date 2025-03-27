import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from skopt import BayesSearchCV

# Load Dataset (Kaggle Credit Card Fraud Dataset)
df = pd.read_csv("creditcard.csv")

# Normalize 'Amount' Feature
df['Amount'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
df.drop(['Time'], axis=1, inplace=True)  # Drop Time column

# Separate Features & Labels
X = df.drop(columns=['Class'])
y = df['Class']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define LightGBM Model
lgb_model = lgb.LGBMClassifier(objective='binary', metric='auc', boosting_type='gbdt', random_state=42)

# Bayesian Hyperparameter Optimization
param_grid = {
    'num_leaves': (20, 150),
    'max_depth': (3, 15),
    'learning_rate': (0.001, 0.3, 'log-uniform'),
    'n_estimators': (50, 500),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0)
}

bayes_search = BayesSearchCV(lgb_model, param_grid, n_iter=30, cv=5, scoring='roc_auc', random_state=42)
bayes_search.fit(X_train, y_train)

# Train Best Model
best_lgb = bayes_search.best_estimator_
best_lgb.fit(X_train, y_train)

# Predictions
y_pred = best_lgb.predict(X_test)
y_prob = best_lgb.predict_proba(X_test)[:, 1]

# Evaluation
print("Best Hyperparameters:", bayes_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))
