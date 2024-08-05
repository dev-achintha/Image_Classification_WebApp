import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

# Set page configuration
st.set_page_config(page_title="Image Classification App", layout="wide")

# Load pre-trained MobileNetV2 model
@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights='imagenet')

model = load_model()

# Function to classify image
def classify_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = np.array(image).astype(np.float32)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    return tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]

# Sidebar
st.sidebar.title("Image Classification")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Create two columns with fixed width
col1, col2 = st.columns([1, 1])

# Function to display classification results
def display_results(predictions, container):
    container.subheader("Classification Results")
    for i, (imagenet_id, label, score) in enumerate(predictions):
        container.write(f"{i + 1}. {label}: {score * 100:.2f}%")
        if score >= confidence_threshold:
            st.balloons()
    
    # Visualization
    container.subheader("Prediction Visualization")
    scores = [score for (imagenet_id, label, score) in predictions]
    labels = [label for (imagenet_id, label, score) in predictions]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(labels, scores)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Confidence Score')
    ax.set_title('Top 5 Predictions')
    container.pyplot(fig)

# Initialize session state
if 'current_image' not in st.session_state:
    st.session_state.current_image = Image.open("sample_image.png").convert('RGB')
    st.session_state.current_predictions = classify_image(st.session_state.current_image)

# Update current image and predictions if a new file is uploaded
if uploaded_file is not None:
    st.session_state.current_image = Image.open(uploaded_file).convert('RGB')
    with st.spinner('Classifying...'):
        # Progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        st.session_state.current_predictions = classify_image(st.session_state.current_image)

# Display current image and predictions
with col1:
    st.image(st.session_state.current_image, caption="Current Image", use_column_width=True)

with col2:
    display_results(st.session_state.current_predictions, col2)

# Additional information
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("This app uses a pre-trained MobileNetV2 model to classify images. Upload an image and adjust the confidence threshold to see the results!")
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Achintha | [GitHub Repository](https://github.com/dev-achintha/Image_Classification_WebApp)")