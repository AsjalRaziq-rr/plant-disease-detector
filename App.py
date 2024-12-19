import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
import streamlit as st
from PIL import Image

# Load the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Build the final model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(15, activation='softmax')  # Update classes for your dataset
])

# Load pre-trained weights (replace with your trained model's weights file)
model.load_weights('/path/to/your/trained/model.h5')

# Class mapping
class_label = {
    0: 'Tomato_healthy',
    1: 'Tomato_bacterial_spot',
    2: 'Tomato_early_blight',
    3: 'Tomato_late_blight',
    4: 'Tomato_leaf_mold',
    5: 'Tomato_septoria_leaf_spot',
    6: 'Tomato_spider_mites',
    7: 'Tomato_target_spot',
    8: 'Tomato_yellow_leaf_curl_virus',
    9: 'Tomato_mosaic_virus',
    10: 'Potato_healthy',
    11: 'Potato_early_blight',
    12: 'Potato_late_blight',
    13: 'Potato_viral_disease',
    14: 'Potato_bacterial_disease'
}  # Adjust classes to match your dataset

# Streamlit app
st.title("Plant Disease Detection")
st.write("Upload an image of a plant leaf to classify its disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption="Uploaded Image", use_column_width=True)
    
    # Process the image for prediction
    img = image_data.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make predictions
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    predicted_class = class_label[class_idx]
    
    st.write(f"### Prediction: **{predicted_class}**")

