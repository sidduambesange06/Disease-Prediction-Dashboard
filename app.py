from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import os
from nlp_processor import SymptomExtractor

app = Flask(__name__)

# Load the trained model
with open('disease_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load the feature names
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Initialize the symptom extractor
symptom_extractor = SymptomExtractor(feature_names)

@app.route('/')
def home():
    return render_template('index.html', symptoms=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        symptoms = data.get('symptoms', {})
        
        # Create a DataFrame with all features set to 0
        input_data = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Set the values for the symptoms that are present
        for symptom, value in symptoms.items():
            if symptom in feature_names:
                input_data[symptom] = value
        
        # Make prediction
        prediction = model.predict(input_data)
        predicted_disease = label_encoder.inverse_transform(prediction)[0]
        
        # Get prediction probability
        probabilities = model.predict_proba(input_data)[0]
        max_probability = max(probabilities) * 100
        
        # Get top 3 predictions with probabilities
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_diseases = label_encoder.inverse_transform(top_indices)
        top_probabilities = [probabilities[i] * 100 for i in top_indices]
        
        top_predictions = [
            {"disease": disease, "probability": prob} 
            for disease, prob in zip(top_diseases, top_probabilities)
        ]
        
        return jsonify({
            'disease': predicted_disease,
            'confidence': f"{max_probability:.2f}%",
            'top_predictions': top_predictions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_free_text', methods=['POST'])
def predict_free_text():
    try:
        # Get data from request
        data = request.json
        symptom_text = data.get('symptom_text', '')
        checkbox_symptoms = data.get('checkbox_symptoms', {})
        
        # Extract symptoms from free text
        extracted_symptoms = symptom_extractor.extract_symptoms(symptom_text)
        
        # Create a DataFrame with all features set to 0
        input_data = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Set the values for the symptoms extracted from text
        for symptom in extracted_symptoms:
            if symptom in feature_names:
                input_data[symptom] = 1
        
        # Add checkbox symptoms if provided
        for symptom, value in checkbox_symptoms.items():
            if symptom in feature_names:
                input_data[symptom] = value
        
        # If no symptoms were found, return an error
        if input_data.sum().sum() == 0:
            return jsonify({
                'error': 'No recognizable symptoms found in the text. Please try again with different wording or use the checkboxes.'
            }), 400
        
        # Make prediction
        prediction = model.predict(input_data)
        predicted_disease = label_encoder.inverse_transform(prediction)[0]
        
        # Get prediction probability
        probabilities = model.predict_proba(input_data)[0]
        max_probability = max(probabilities) * 100
        
        # Get top 3 predictions with probabilities
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_diseases = label_encoder.inverse_transform(top_indices)
        top_probabilities = [probabilities[i] * 100 for i in top_indices]
        
        top_predictions = [
            {"disease": disease, "probability": prob} 
            for disease, prob in zip(top_diseases, top_probabilities)
        ]
        
        # Return the identified symptoms along with the prediction
        return jsonify({
            'disease': predicted_disease,
            'confidence': f"{max_probability:.2f}%",
            'top_predictions': top_predictions,
            'identified_symptoms': extracted_symptoms
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({'symptoms': feature_names})

if __name__ == '__main__':
    app.run(debug=True)