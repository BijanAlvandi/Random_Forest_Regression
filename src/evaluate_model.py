import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Load artifacts
artifacts = joblib.load('../Model/model_artifacts.joblib')
model = artifacts['model']
feature_names = artifacts['feature_names']

# Load test data
test_data = pd.read_csv('Model/test_data.csv')
X_test = test_data[feature_names]
y_test = test_data['MedHouseVal']

# Evaluate
y_pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")