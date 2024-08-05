import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.applications.MobileNetV2(weights='imagenet')

st.sidebar.title("Image Classification")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

if uploaded_file is not None:
    with st.spinner('Classifying...'):
        # Convert image to numpy array and ensure it is writable
        image = np.array(image)
        image.setflags(write=1)
        
        # Preprocess the image
        image = tf.image.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        
        # Predict using the model
        predictions = model.predict(image)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)
        
        st.success('Classification complete!')
        
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
            st.write(f"{i + 1}. {label}: {score * 100:.2f}%")
            
            if score >= confidence_threshold:
                st.balloons()

if uploaded_file is not None:
    scores = [score for (imagenet_id, label, score) in decoded_predictions[0]]
    labels = [label for (imagenet_id, label, score) in decoded_predictions[0]]
    
    fig, ax = plt.subplots()
    ax.barh(labels, scores)
    ax.set_xlim(0, 1)
    st.pyplot(fig)