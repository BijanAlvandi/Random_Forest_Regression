import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Load artifacts
artifacts = joblib.load('E:\\Python_projects\\Machine_Learning\\10_Random_Forest_Regression\\Model\\trained_model.joblib')
model = artifacts['model']
feature_names = artifacts['feature_names']

# Load test data
test_data = pd.read_csv('E:\\Python_projects\\Machine_Learning\\10_Random_Forest_Regression\\Model\\test_data.csv')
X_test = test_data[feature_names]
y_test = test_data['MedHouseVal']

# Evaluate
y_pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
