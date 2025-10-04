#import the libary 
import streamlit as st
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch

# Streamlit configration
st.set_page_config(
    page_title="Fashion Item Classifier",
    page_icon="👕",
    layout="centered"
)

# Load model and processor
model_name = "patrickjohncyh/fashion-clip"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#streamlit cover(front-end)
st.title("👕 Fashion Item Classifier")
st.write("Upload an image of a clothing item and the app will predict its class!")

# Image uploader
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # streamlit button
    if st.button("Classify"):
        with st.spinner("Predicting..."):
            try:
                # Preprocess image and text
                inputs = processor(text=class_names, images=image, return_tensors="pt", padding=True)

                with torch.no_grad():
                    # Image-text similarity scores
                    logits_per_image = model(**inputs).logits_per_image  
                     # Convert to probabilities
                    probs = logits_per_image.softmax(dim=1) 

                # Get predicted class
                predicted_class = class_names[torch.argmax(probs)]
                st.success(f"✅ Prediction: **{predicted_class}**")

                # Show probabilities
                st.write("Prediction probabilities:")
                for i, class_name in enumerate(class_names):
                    st.write(f"{class_name}: {probs[0][i]:.4f}")
            #if any error in prediciton
            except Exception as e:
                st.error(f"❌ Error in prediction: {e}")
