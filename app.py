import streamlit as st
import joblib
import gdown

# Title
st.set_page_config(page_title="AI Doctor", page_icon="🩺")
st.title("🤖 AI Doctor - Disease Prediction from Symptoms")

# Description
st.markdown("Enter your symptoms and let the AI doctor predict the possible disease.")

# Load model from Google Drive (if not already downloaded)
model_url = "https://drive.google.com/uc?id=1cmDxvkDxU5CyLKHUyiB29YMJyIc0HQFy"
output_path = "disease_prediction_model.pkl"
gdown.download(model_url, output_path, quiet=False)

# Load model
model = joblib.load("disease_prediction_model.pkl")

# Define symptom options
symptoms_list = [
    'abdominal_pain', 'acidity', 'back_pain', 'chest_pain', 'chills',
    'constipation', 'cough', 'depression', 'diarrhoea', 'dizziness',
    'fatigue', 'fever', 'headache', 'joint_pain', 'loss_of_appetite',
    'muscle_pain', 'nausea', 'skin_rash', 'sore_throat', 'vomiting',
    'weight_gain', 'weight_loss'
]

# Symptom multiselect
selected_symptoms = st.multiselect("🩹 Select your symptoms:", symptoms_list)

# Predict button
if st.button("Predict Disease"):
    if selected_symptoms:
        input_symptoms = ", ".join(selected_symptoms)
        prediction = model.predict([[input_symptoms]])[0]  # ✅ FIXED: 2D input
        st.success(f"🧾 **Predicted Disease:** {prediction}")
    else:
        st.warning("⚠️ Please select at least one symptom.")
