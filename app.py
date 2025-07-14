import streamlit as st
import joblib
import numpy as np
import gdown

# -------------------------------
# Download the model from Google Drive (if not already)
# -------------------------------
MODEL_PATH = "disease_prediction_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

model_url = "https://drive.google.com/uc?id=1cmDxvkDxU5CyLKHUyiB29YMJyIc0HQFy"
gdown.download(model_url, MODEL_PATH, quiet=False)

# -------------------------------
# Load files
# -------------------------------
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# -------------------------------
# Hardcoded Label Map (Example)
# -------------------------------
label_map = {
    0: 'Fungal infection',
    1: 'Allergy',
    2: 'GERD',
    3: 'Chronic cholestasis',
    4: 'Drug Reaction',
    5: 'Peptic ulcer disease',
    6: 'AIDS',
    7: 'Diabetes',
    8: 'Gastroenteritis',
    9: 'Bronchial Asthma',
    10: 'Hypertension',
    11: 'Migraine',
    12: 'Cervical spondylosis',
    13: 'Paralysis (brain hemorrhage)',
    14: 'Jaundice',
    15: 'Malaria',
    16: 'Chicken pox',
    17: 'Dengue',
    18: 'Typhoid',
    19: 'Hepatitis A',
    20: 'Hepatitis B',
    21: 'Hepatitis C',
    22: 'Hepatitis D',
    23: 'Hepatitis E',
    24: 'Alcoholic hepatitis',
    25: 'Tuberculosis',
    26: 'Common Cold',
    27: 'Pneumonia',
    28: 'Heart attack',
    29: 'Varicose veins',
    30: 'Hypothyroidism',
    31: 'Hyperthyroidism',
    32: 'Hypoglycemia',
    33: 'Osteoarthritis',
    34: 'Arthritis',
    35: '(vertigo) Paroxysmal Positional Vertigo',
    36: 'Acne',
    37: 'Urinary tract infection',
    38: 'Psoriasis',
    39: 'Impetigo'
}

# -------------------------------
# Streamlit App
# -------------------------------
st.title("🩺 Symptom-based Disease Prediction")

user_input = st.text_input("Enter your symptoms (comma-separated):", "")

if user_input:
    input_symptoms = [sym.strip().lower() for sym in user_input.split(',')]
    input_text = " ".join(input_symptoms)
    input_vector = vectorizer.transform([input_text])
    prediction = model.predict(input_vector)[0]
    disease = label_map.get(prediction, "Unknown Disease")

    st.success(f"**Predicted Disease:** {disease}")
