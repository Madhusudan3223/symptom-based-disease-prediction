import streamlit as st
import joblib
import gdown
import os
import numpy as np

# -------------------------------
# File paths and Google Drive model download
# -------------------------------

MODEL_PATH = "disease_prediction_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
LABELMAP_PATH = "label_map.pkl"

# Download model file from Google Drive if not already present
model_file_id = "1cmDxvkDxU5CyLKHUyiB29YMJyIc0HQFy"
model_url = f"https://drive.google.com/uc?id={model_file_id}"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    gdown.download(model_url, MODEL_PATH, quiet=False)

# -------------------------------
# Load files
# -------------------------------

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
label_map = joblib.load(LABELMAP_PATH)

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="AI Doctor Bot", page_icon="🩺")
st.title("🩺 Symptom-based Disease Prediction")

symptoms = st.text_area("Enter your symptoms (comma-separated):", placeholder="e.g. headache, fever, chills")

if st.button("Predict Disease"):
    if symptoms.strip() == "":
        st.warning("Please enter symptoms to get a prediction.")
    else:
        # Preprocess and predict
        symptoms_cleaned = [symptoms]
        X = vectorizer.transform(symptoms_cleaned)
        prediction = model.predict(X)[0]
        disease = label_map.get(prediction, "Unknown Disease")

        st.success(f"Predicted Disease: **{disease}**")
