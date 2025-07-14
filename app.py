import streamlit as st
import joblib
import requests
import os

# -------------------------------
# Function to download from Google Drive
# -------------------------------
def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# -------------------------------
# Download model from Google Drive if not present
# -------------------------------
MODEL_PATH = "disease_prediction_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
LABELMAP_PATH = "label_map.pkl"

MODEL_FILE_ID = "1cmDxvkDxU5CyLKHUyiB29YMJyIc0HQFy"  # Update if changed
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    download_file_from_google_drive(MODEL_FILE_ID, MODEL_PATH)

# -------------------------------
# Load files
# -------------------------------
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
label_map = joblib.load(LABELMAP_PATH)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🩺 AI Doctor Bot - Disease Prediction")

st.markdown("Enter your symptoms below (e.g., *headache, fever, cough*) to predict possible disease.")

user_input = st.text_area("🔍 Enter Symptoms", placeholder="e.g. chills, vomiting, high fever, sweating")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some symptoms.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        disease_name = label_map.get(prediction, "Unknown Disease")
        st.success(f"🧠 Predicted Disease: **{disease_name}**")
