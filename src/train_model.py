import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_california_housing
from joblib import dump
import os

# Create model directory
save_dir = '../Model'
os.makedirs(save_dir, exist_ok=True)

# Load dataset
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Save test set separately for evaluation script
test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_csv(os.path.join(save_dir, 'test_data.csv'), index=False)

# Initialize base model
base_model = RandomForestRegressor(
    random_state=42,
    n_jobs=-1  # Utilize all cores
)

# Hyperparameter tuning configuration
param_grid = {
    'n_estimators': [10, 20],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Setup GridSearchCV
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,  # Parallel processing
    verbose=2,   # Detailed progress logs
    refit=True   # Refit best model on entire training set
)

# Execute hyperparameter search
print("Starting hyperparameter tuning...")
grid_search.fit(X_train, y_train)
print("Tuning completed!")

# Save best model and metadata
dump(
    {
        'model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'feature_names': list(X.columns),
        'cv_results': grid_search.cv_results_
    },
    os.path.join(save_dir, 'trained_model.joblib')
)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Model saved to {save_dir}/trained_model.joblib")
print(f"Test data saved to {save_dir}/test_data.csv")
