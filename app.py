import streamlit as st
import joblib
import numpy as np
import re
import os
import gdown

# -------------------------------
# Download model from Google Drive if not present
# -------------------------------
MODEL_PATH = "disease_prediction_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
MODEL_FILE_ID = "1cmDxvkDxU5CyLKHUyiB29YMJyIc0HQFy"  

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False)

# -------------------------------
# Load model and vectorizer
# -------------------------------
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# -------------------------------
# Hardcoded label_map dictionary
# -------------------------------
label_map = {
    0: "Fungal infection",
    1: "Allergy",
    2: "GERD",
    3: "Chronic cholestasis",
    4: "Drug Reaction",
    5: "Peptic ulcer disease",
    6: "AIDS",
    7: "Diabetes",
    8: "Gastroenteritis",
    9: "Bronchial Asthma",
    10: "Hypertension",
    11: "Migraine",
    12: "Cervical spondylosis",
    13: "Paralysis (brain hemorrhage)",
    14: "Jaundice",
    15: "Malaria",
    16: "Chicken pox",
    17: "Dengue",
    18: "Typhoid",
    19: "Hepatitis A",
    20: "Hepatitis B",
    21: "Hepatitis C",
    22: "Hepatitis D",
    23: "Hepatitis E",
    24: "Alcoholic hepatitis",
    25: "Tuberculosis",
    26: "Common Cold",
    27: "Pneumonia",
    28: "Dimorphic hemorrhoids(piles)",
    29: "Heart attack",
    30: "Varicose veins",
    31: "Hypothyroidism",
    32: "Hyperthyroidism",
    33: "Hypoglycemia",
    34: "Osteoarthristis",
    35: "Arthritis",
    36: "(Vertigo) Paroymsal Positional Vertigo",
    37: "Acne",
    38: "Urinary tract infection",
    39: "Psoriasis",
    40: "Impetigo"
}

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="🩺 Disease Predictor", page_icon="🩺")
st.title("🩺 Symptom-based Disease Prediction")
st.write("Enter your symptoms separated by commas (e.g., `fever, chills, headache`).")

user_input = st.text_input("Symptoms:")

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter at least one symptom.")
    else:
        # Preprocess input
        symptoms = re.sub(r'\s+', ' ', user_input.strip())
        X_input = vectorizer.transform([symptoms])
        prediction = model.predict(X_input)
        predicted_label = prediction[0]
        predicted_disease = label_map.get(predicted_label, "Unknown Disease")
        
        st.success(f"🧬 Predicted Disease: **{predicted_disease}**")
