import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('disease_symptoms_dataset.csv')
print(f"Dataset shape: {df.shape}")

# Display the first few rows
print("\nFirst few rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Get unique diseases
diseases = df['disease'].unique()
print(f"\nNumber of unique diseases: {len(diseases)}")
print(f"Unique diseases: {diseases[:5]}... (showing first 5)")

# Preprocess the data
print("\nPreprocessing data...")
# Assuming the dataset has columns for symptoms and a 'disease' column for the target
X = df.drop('disease', axis=1)
y = df['disease']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the label encoder for later use
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Train a Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.4f}")

# Save the model
print("\nSaving model...")
with open('disease_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the feature names for later use
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("\nModel training complete!")