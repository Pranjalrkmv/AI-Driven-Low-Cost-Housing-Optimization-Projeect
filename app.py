from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Step 1: Initialize Flask App
app = Flask(__name__)

# Step 2: Load the Saved Machine Learning Models
cost_model = joblib.load('models/cost_prediction_model.pkl')  # Cost prediction model
material_model = joblib.load('models/material_recommendation_model.pkl')  # Material recommendation model

# Step 3: Home Route - Just to Check if the API is Running
@app.route('/')
def home():
    return render_template('index.html')

# Step 4: API Route for Cost Prediction
@app.route('/predict_cost', methods=['POST'])
def predict_cost():
    try:
        # Get the input data from the POST request
        data = request.get_json()

        # Prepare the features (make sure the keys match the input data)
        features = np.array([[data['Durability'],
                              data['Climate Suitability'],
                              data['material_Brick'],
                              data['material_Cement'],
                              data['material_Steel'],
                              data['material_Wood']]])

        # Use the model to make a prediction
        cost_prediction = cost_model.predict(features)

        # Return the result as JSON
        return jsonify({'predicted_cost': cost_prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

# Step 5: API Route for Material Recommendation
@app.route('/predict_material', methods=['POST'])
def predict_material():
    try:
        # Get the input data from the POST request
        data = request.get_json()

        # Prepare the features
        features = np.array([[data['Durability'],
                              data['Climate Suitability']]])

        # Use the model to make a prediction
        material_classes = ['Brick', 'Cement', 'Steel', 'Wood']
        material_prediction = material_model.predict(features)

        # Return the predicted material
        predicted_material = material_classes[material_prediction[0]]
        return jsonify({'predicted_material': predicted_material})

    except Exception as e:
        return jsonify({'error': str(e)})

# Step 6: Run the Flask App
if __name__ == '__main__':
    app.run(debug=True)
