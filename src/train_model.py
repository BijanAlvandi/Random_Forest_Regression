import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import GridSearchCV
from joblib import dump
import os

save_dir = 'Model'
os.makedirs(save_dir, exist_ok=True)

# Loading the dataset
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the mode
regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
)

regressor.fit(X_train, y_train)

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 30],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Saving the model
dump({
    'model': best_model,
    'X_test': X_test,
    'y_test': y_test,

}, os.path.join(save_dir, 'model_artifacts.joblib'))
