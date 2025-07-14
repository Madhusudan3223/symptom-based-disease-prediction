import streamlit as st
import joblib
import numpy as np
import gdown
import os

# Download model from Google Drive if not already present
MODEL_URL = "https://drive.google.com/uc?id=1cmDxvkDxU5CyLKHUyiB29YMJyIc0HQFy"
MODEL_PATH = "disease_prediction_model.pkl"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Title
st.title("🩺 Symptom-based Disease Prediction")

# List of symptoms
symptom_list = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing',
    'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue',
    'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_urination', 'fatigue',
    'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss',
    'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level',
    'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating',
    'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine',
    'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
    'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes'
]

# Multiselect input
selected_symptoms = st.multiselect(
    "🔍 Select your symptoms:",
    options=symptom_list
)

# Prediction
if st.button("Predict Disease"):
    if selected_symptoms:
        input_symptoms = ", ".join(selected_symptoms)
        prediction = model.predict([input_symptoms])[0]
        st.success(f"🧾 **Predicted Disease:** {prediction}")
    else:
        st.warning("⚠️ Please select at least one symptom.")
