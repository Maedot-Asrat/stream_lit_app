import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Load your trained model
model = load_model('plant_disease_model.h5')

# Define class names
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Barley_blight', 'Corn-Common_rust')

# Page config for wider layout & title
st.set_page_config(page_title="Arada Pest Detection", layout="wide")


st.markdown(
    """
    <style>
     .css-18e3th9, .css-1d391kg {
        background-color: white !important;
        color: black !important;
    }
    /* White background for the whole app */
    .reportview-container, .main {
        background-color: white;
    }

    /* Style the predict button */
    div.stButton > button:first-child {
        background-color: #90ee90;
        color: black;
        font-weight: bold;
        height: 3em;
        width: 100%;
        border-radius: 8px;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #7ccd7c;
        color: white;
    }

    /* Style the file uploader label */
    .stFileUploader label {
        font-weight: bold;
        color: #2e7d32;
    }

    /* Style file uploader dropzone */
    .stFileUploader div[role="button"] {
        background-color: #d0f0c0;
        border-radius: 8px;
        padding: 1rem;
        border: 2px dashed #90ee90;
        color: #2e7d32;
    }
    .stFileUploader div[role="button"]:hover {
        background-color: #b0dca0;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header section with new title and styling
st.markdown(
    """
    <h1 style='text-align: center; color: #2e7d32; font-weight: 900; margin-bottom: 0.1em;'>
        🌿 Arada Pest Detection 🌿
    </h1>
    <p style='text-align: center; font-size: 18px; color: #4a7f42;'>
        Upload a plant leaf image, and our AI model will detect the disease.
    </p>
    """,
    unsafe_allow_html=True,
)

# Two column layout: image upload & prediction result
col1, col2 = st.columns([1, 1])

with col1:
    plant_image = st.file_uploader("Choose a plant leaf image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    submit = st.button('Predict Disease')

with col2:
    st.empty()  # Placeholder for showing image & prediction

if submit:
    if plant_image is None:
        st.warning("Please upload an image before clicking predict.")
    else:
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Show uploaded image on right column
        with col2:
            st.image(opencv_image, channels="BGR", caption="Uploaded Leaf Image", use_column_width=True)

        # Preprocess image for model prediction
        img_resized = cv2.resize(opencv_image, (256, 256))
        img_expanded = np.expand_dims(img_resized, axis=0)

        # Predict disease class
        preds = model.predict(img_expanded)
        pred_index = np.argmax(preds)
        pred_class = CLASS_NAMES[pred_index]

        # Parse class for pretty output
        plant, disease = pred_class.split('-')

        # Show prediction result with styling
        with col2:
            st.markdown(
                f"<h2 style='color: red;'>Prediction:</h2>"
                f"<h3>{plant} leaf is affected by <b>{disease.replace('_', ' ')}</b></h3>",
                unsafe_allow_html=True,
            )
            st.info(f"Confidence: {preds[0][pred_index]:.2%}")
