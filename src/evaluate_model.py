import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Load artifacts
artifacts = joblib.load('Model/model_artifacts.joblib')
model = artifacts['model']
feature_names = artifacts['feature_names']

# Load test data
test_data = pd.read_csv()