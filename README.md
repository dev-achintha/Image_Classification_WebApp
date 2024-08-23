# Image Classification Web App

This repository contains the code for a Streamlit web application that uses a pre-trained MobileNetV2 model to classify images. The app allows users to upload an image, apply filters, adjust the confidence threshold, and visualize the classification results with a heatmap.

## Demo

Check out the live demo of the web app <h2>[👉here👈](https://imageclassificationwebapp-ulgcx4shjvwykzf8fzgqee.streamlit.app/)</h2>.

## Features

- Upload an image (jpg, jpeg, png)
- Apply image filters (Grayscale, Edge Detection)
- Adjust the confidence threshold for classification results
- View the top 10 predicted classes with confidence scores
- Visualize the predictions using a horizontal bar chart
- Display a Class Activation Map (heatmap) to highlight regions influencing the model's prediction

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
import seaborn as sns
import time
import cv2

# LOAD PRE-TRAINED MOBILENETV2 MODEL
@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights='imagenet')

model = load_model()

# SIDEBAR FOR FILE UPLOAD, FILTERS, AND PARAMETERS
st.sidebar.title("Image Classification")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
show_heatmap = st.sidebar.checkbox("Show Heatmap", value=False)
apply_filter = st.sidebar.selectbox("Apply Filter", ["None", "Grayscale", "Edge Detection"])

# MAIN CONTENT
st.title("Image Classification with MobileNetV2")

# DISPLAY UPLOADED IMAGE AND CLASSIFICATION RESULTS
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # APPLY SELECTED FILTER
    if apply_filter == "Grayscale":
        image = image.convert('L').convert('RGB')
    elif apply_filter == "Edge Detection":
        img_array = np.array(image)
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        img_edges = cv2.Canny(img_gray, 100, 200)
        image = Image.fromarray(cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB))

    st.image(image, caption='Uploaded Image', use_column_width=True)

    # CLASSIFY IMAGE AND SHOW PROGRESS
    with st.spinner('Classifying...'):
        image_resized = np.array(image).astype(np.float32)
        image_resized = tf.image.resize(image_resized, (224, 224))
        image_resized = image_resized / 255.0  # NORMALIZE TO [0,1]
        image_resized = np.expand_dims(image_resized, axis=0)
        predictions = model.predict(image_resized)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=10)[0]

    # DISPLAY RESULTS
    st.subheader("Classification Results:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        if score >= confidence_threshold:
            st.write(f"{i + 1}. {label}: {score * 100:.2f}%")

    # VISUALIZE PREDICTIONS
    st.subheader("Prediction Visualization:")
    scores = [score for (imagenet_id, label, score) in decoded_predictions]
    labels = [label for (imagenet_id, label, score) in decoded_predictions]
    fig, ax = plt.subplots()
    sns.barplot(x=scores, y=labels, ax=ax)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Confidence Score')
    ax.set_title('Top 10 Predictions')
    st.pyplot(fig)

    # DISPLAY HEATMAP IF SELECTED
    if show_heatmap:
        st.subheader("Class Activation Map")
        heatmap = make_gradcam_heatmap(
            np.expand_dims(np.array(image.resize((224, 224))), axis=0),
            model,
            last_conv_layer_name='Conv_1'
        )
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.imshow(heatmap, cmap='jet', alpha=0.5)  # OVERLAY HEATMAP
        plt.axis('off')
        st.pyplot(plt)

else:
    st.write("Please upload an image to get started!")

# INFORMATION ABOUT THE PROJECT
st.sidebar.markdown("---")
st.sidebar.write("This app uses a pre-trained MobileNetV2 model to classify images.")
st.sidebar.write("Upload an image, apply filters, adjust the confidence threshold, and view the heatmap to see the results!")
```
