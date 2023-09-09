# app.py

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load models
gender_model = tf.keras.models.load_model('gender_model.h5')
age_model = tf.keras.models.load_model('age_model.h5')

def predict_age_and_gender(image):
    # Pre-process image to fit model input
    img = image.resize((48, 48)).convert('L')  # Grayscale conversion
    img_arr = np.array(img).astype('float32') / 255.0
    img_arr = np.expand_dims(img_arr, axis=[0, -1])

    # Predictions
    predicted_age = age_model.predict(img_arr)
    predicted_gender_prob = gender_model.predict(img_arr)
    
    gender = "Male" if predicted_gender_prob[0][0] < 0.5 else "Female"

    return int(predicted_age[0][0]), gender

st.title("Age and Gender Prediction App")

uploaded_image = st.file_uploader("Choose an image...", type="jpg")

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Predicting...")
    
    age, gender = predict_age_and_gender(image)
    
    st.write(f"Predicted Age: {age}")
    st.write(f"Predicted Gender: {gender}")

