import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# 🔃 Load model
model = tf.keras.models.load_model("crop_disease_model.h5")

# 🌿 Define class names (replace these with your actual class names)
class_names = ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold',
               'Septoria Leaf Spot', 'Spider Mites', 'Target Spot', 'Yellow Leaf Curl Virus',
               'Mosaic Virus', 'Healthy']

# 🖼️ Streamlit app UI
st.set_page_config(page_title="CropGuardX - Tomato Leaf Diagnosis", layout="centered")
st.title("🍅 CropGuardX - Tomato Plant Disease Detection")
st.markdown("Upload an image of a tomato leaf to detect disease and get suggestions 🌱")

# 📤 File uploader
uploaded_file = st.file_uploader("Upload Tomato Leaf Image", type=["jpg", "jpeg", "png"])

# 🧠 Prediction function
def predict_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))           # Must match model input shape
    img = img / 255.0                           # Normalize
    img = np.expand_dims(img, axis=0)          # Add batch dimension
    predictions = model.predict(img)
    confidence = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class, confidence

# 🚀 Predict on Upload
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Analyzing...")

    # 🎯 Prediction
    label, confidence = predict_image(image)

    st.success(f"**Prediction:** {label}")
    st.info(f"**Confidence:** {confidence*100:.2f}%")

    # 💡 Health tips
    if label.lower() != "healthy":
        st.warning("⚠️ This plant might be infected. Please consult an agricultural expert.")
    else:
        st.balloons()
        st.success("✅ This leaf looks healthy! Keep it up!")

