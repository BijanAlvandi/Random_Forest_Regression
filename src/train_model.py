import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_california_housing
from joblib import dump
import os

# Setup
save_dir = '../Model'
os.makedirs(save_dir, exist_ok=True)

# Load data
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [10, 20, 30],
    'max_depth': [None, 10, 30],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Save model and test data
dump({
    'model': best_model,
    'X_test': X_test,
    'y_test': y_test,
    'best_params': grid_search.best_params_,
}, os.path.join(save_dir, 'model_artifacts.joblib'))
