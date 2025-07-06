import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from joblib import dump

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
    random_state = 42,
)
regressor.fit(X_train, y_train)

# save the trained model
dump(regressor, 'models/random_forest_model.joblib')

