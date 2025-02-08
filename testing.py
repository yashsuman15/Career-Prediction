import streamlit as st
import numpy as np
import joblib
from scipy import sparse
import pickle
import pandas as pd

def preprocess_data(new_data_array, encoder, scaler):
    """Preprocesses the new data array using the fitted encoder and scaler."""
    # One-Hot Encoding
    encoded_data = encoder.transform(new_data_array)

    # Scaling
    scaled_data = scaler.transform(encoded_data)

    # Conversion to Sparse Matrix
    preprocessed_data = sparse.csr_matrix(scaled_data)

    return preprocessed_data

# Load the trained model
model_path = 'final_model.pkl'  # Update with your actual model file
clf = joblib.load(model_path)
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# Test Array 1
test_array_1 = pd.DataFrame([
    [70, 95, 65, 80, 90, 80, 95, 75, 65, 10, 8, 5, 7, 7, 'yes', 'yes', 'yes',
     'machine learning', 'data science', 'no', 'no', 'excellent', 'medium',
     'Management', 'Business process analyst', 'higherstudies', 'Product based',
     'yes', 'no', 'Mystery', 'salary', 'yes', 'gentle', 'Management', 'salary',
     'smart worker', 'yes', 'yes']
])

# Test Array 2
test_array_2 = pd.DataFrame([
    [60, 85, 70, 75, 80, 75, 90, 70, 70, 8, 6, 3, 5, 5, 'no', 'yes', 'no',
     'programming', 'web development', 'yes', 'no', 'good', 'high',
     'Technical', 'Software Developer', 'bachelors', 'Service based',
     'no', 'yes', 'Comedy', 'work life balance', 'no', 'aggressive', 'Technical',
     'salary', 'hard worker', 'no', 'yes']
])

# Test Array 3
test_array_3 = pd.DataFrame([
    [65, 90, 68, 78, 85, 78, 92, 72, 68, 9, 7, 4, 6, 6, 'yes', 'no', 'yes',
     'data analysis', 'cloud computing', 'no', 'yes', 'excellent', 'low',
     'Management', 'Data Analyst', 'masters', 'Product based',
     'yes', 'no', 'Thriller', 'growth', 'yes', 'gentle', 'Management', 'salary',
     'smart worker', 'yes', 'no']
])

# Preprocess the test arrays
preprocessed_test_1 = preprocess_data(test_array_1, encoder, scaler)
preprocessed_test_2 = preprocess_data(test_array_2, encoder, scaler)
preprocessed_test_3 = preprocess_data(test_array_3, encoder, scaler)

# Make predictions
prediction_1 = clf.predict(preprocessed_test_1)
prediction_2 = clf.predict(preprocessed_test_2)
prediction_3 = clf.predict(preprocessed_test_3)

print("Prediction for Test Array 1:", prediction_1)
print("Prediction for Test Array 2:", prediction_2)
print("Prediction for Test Array 3:", prediction_3)