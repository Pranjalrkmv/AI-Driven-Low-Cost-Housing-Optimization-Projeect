import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib

# Load the cleaned dataset
data = pd.read_csv('cleaned_housing_data.csv')

# Step 1: Cost Prediction (Regression)
# Features: All except 'Cost', Target: 'Cost'
X_cost = data.drop('Cost', axis=1)
y_cost = data['Cost']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cost, y_cost, test_size=0.2, random_state=42)

# Train a regression model
cost_model = LinearRegression()
cost_model.fit(X_train_c, y_train_c)

# Save the regression model
joblib.dump(cost_model, 'cost_prediction_model.pkl')

# # Step 2: Material Recommendation (Classification)
# # Features: All except 'Material', Target: 'Material'
# X_material = data.drop('Material', axis=1)
# y_material = data['Material']

# X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_material, y_material, test_size=0.2, random_state=42)

# # Train a classification model
# material_model = RandomForestClassifier(n_estimators=100, random_state=42)
# material_model.fit(X_train_m, y_train_m)

# # Save the classification model
# joblib.dump(material_model, 'material_recommendation_model.pkl')

# print("Models trained and saved successfully!")
# Step 2: Material Recommendation (Classification)
# Features: All except material columns, Target: one of the material columns
material_columns = ['material_Brick', 'material_Cement', 'material_Steel', 'material_Wood']

# Separate the target (Material) and features
X_material = data.drop(material_columns, axis=1)  # Drop material columns for features
y_material = data[material_columns]  # The material columns as the target

# Split data into training and testing
from sklearn.model_selection import train_test_split
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_material, y_material, test_size=0.2, random_state=42)

# Train a classification model (Random Forest)
from sklearn.ensemble import RandomForestClassifier
material_model = RandomForestClassifier(n_estimators=100, random_state=42)
material_model.fit(X_train_m, y_train_m)

# Save the trained model
import joblib
joblib.dump(material_model, 'material_recommendation_model.pkl')

print("Material Recommendation Model trained and saved successfully!")


