# Image Classification Web App

This repository contains the code for a Streamlit web application that uses a pre-trained MobileNetV2 model to classify images. The app allows users to upload an image and adjust the confidence threshold for the classification results.

## Demo

Check out the live demo of the web app <h2>[👉here👈](https://imageclassificationwebapp-ulgcx4shjvwykzf8fzgqee.streamlit.app/)</h2>.

## Features

- Upload an image (jpg, jpeg, png)
- Adjust the confidence threshold for classification results
- View the top 5 predicted classes with confidence scores
- Visualize the predictions using a horizontal bar chart

## Installation

To run this web app locally, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/image-classification-web-app.git
    cd image-classification-web-app
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

## Code Overview

Here's an overview of the code:

```python
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load pre-trained MobileNetV2 model
@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights='imagenet')

model = load_model()

# Sidebar for file upload and parameters
st.sidebar.title("Image Classification")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Main content
st.title("Image Classification with MobileNetV2")

# Display uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict and show progress
    with st.spinner('Classifying...'):
        # Convert image to numpy array and preprocess
        image = np.array(image).astype(np.float32)
        image = tf.image.resize(image, (224, 224))
        image = image / 255.0  # Normalize to [0,1]
        image = np.expand_dims(image, axis=0)

        # Predict using the model
        predictions = model.predict(image)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]

    # Display results
    st.subheader("Classification Results:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        st.write(f"{i + 1}. {label}: {score * 100:.2f}%")
        if score >= confidence_threshold:
            st.balloons()

    # Visualize predictions
    st.subheader("Prediction Visualization:")
    scores = [score for (imagenet_id, label, score) in decoded_predictions]
    labels = [label for (imagenet_id, label, score) in decoded_predictions]
    fig, ax = plt.subplots()
    ax.barh(labels, scores)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Confidence Score')
    ax.set_title('Top 5 Predictions')
    st.pyplot(fig)

else:
    st.write("Please upload an image to get started!")

# Information about the project
st.sidebar.markdown("---")
st.sidebar.write("This app uses a pre-trained MobileNetV2 model to classify images.")
st.sidebar.write("Upload an image and adjust the confidence threshold to see the results!")
