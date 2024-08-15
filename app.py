import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import cv2

st.set_page_config(page_title="Image Classification App", layout="wide")

# load model
@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights='imagenet')

model = load_model()

# to classify image
def classify_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = np.array(image).astype(np.float32)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    return tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=10)[0]

# Sidebar
st.sidebar.title("Image Classification")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

show_heatmap = st.sidebar.checkbox("Show Heatmap", value=False)
apply_filter = st.sidebar.selectbox("Apply Filter", ["None", "Grayscale", "Edge Detection"])

col1, col2 = st.columns([1, 1])

# to display classification results
def display_results(predictions, container, threshold):
    container.subheader("Classification Results")
    filtered_predictions = [pred for pred in predictions if pred[2] >= threshold]
    
    if not filtered_predictions:
        container.warning(f"No predictions meet the confidence threshold of {threshold:.2f}")
    else:
        for i, (imagenet_id, label, score) in enumerate(filtered_predictions):
            container.write(f"{i + 1}. {label}: {score * 100:.2f}%")
    
    # to visualize
    container.subheader("Prediction Visualization")
    scores = [score for (_, _, score) in predictions]
    labels = [label for (_, label, _) in predictions]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=scores, y=labels, ax=ax)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Confidence Score')
    ax.set_title('Top 10 Predictions')
    ax.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
    ax.legend()
    container.pyplot(fig)

if 'current_image' not in st.session_state:
    st.session_state.current_image = Image.open("sample_image.png").convert('RGB')
    st.session_state.current_predictions = classify_image(st.session_state.current_image)

# update current image and predictions if a new file is uploaded
if uploaded_file is not None:
    st.session_state.current_image = Image.open(uploaded_file).convert('RGB')
    with st.spinner('Classifying...'):
        # Progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        st.session_state.current_predictions = classify_image(st.session_state.current_image)

# apply filter if selected
filtered_image = st.session_state.current_image.copy()
if apply_filter == "Grayscale":
    filtered_image = filtered_image.convert('L').convert('RGB')
elif apply_filter == "Edge Detection":
    img_array = np.array(filtered_image)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 200)
    filtered_image = Image.fromarray(cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB))

# to create heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# display current image and predictions
with col1:
    st.image(filtered_image, caption="Current Image", use_column_width=True)
    
    if show_heatmap:
        st.subheader("Class Activation Map")
        heatmap = make_gradcam_heatmap(
            np.expand_dims(np.array(st.session_state.current_image.resize((224, 224))), axis=0),
            model,
            last_conv_layer_name='Conv_1'
        )

        # display heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(st.session_state.current_image)
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.axis('off')
        st.pyplot(plt)

with col2:
    display_results(st.session_state.current_predictions, col2, confidence_threshold)

st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("This app uses a pre-trained MobileNetV2 model to classify images. Upload an image, adjust the confidence threshold, apply filters, and view the heatmap to explore the results!")
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Achintha | [GitHub Repository](https://github.com/dev-achintha/Image_Classification_WebApp)")
