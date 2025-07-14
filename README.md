🌿 CropGuardX: AI-Powered Tomato Leaf Disease Detection System
📌 Project Description
CropGuardX is an intelligent AI/ML-based solution designed to assist farmers and agricultural professionals in early detection and diagnosis of tomato plant diseases using deep learning and computer vision.

This project uses a Convolutional Neural Network (CNN) trained on the PlantVillage dataset to classify tomato leaf images into various disease categories and suggest appropriate treatments or actions.

🎯 Key Features
✅ Image-based Disease Detection
Upload or capture an image of a tomato leaf, and the system instantly identifies if it’s healthy or affected by a specific disease.

🔍 High Accuracy CNN Model
Trained on thousands of labeled tomato leaf images for high precision.

📸 User-Friendly Interface (Streamlit)
Easy-to-use web interface built using Streamlit for real-time prediction.

💡 Actionable Suggestions
Provides treatment suggestions or preventive measures for detected diseases.

📁 Dataset Used
Name: PlantVillage Tomato Leaf Dataset

Size: ~600 MB

Classes: Includes multiple disease types like:

Tomato Bacterial Spot

Tomato Late Blight

Tomato Leaf Mold

Tomato Yellow Leaf Curl Virus

Tomato Mosaic Virus

...and more

🧠 Model Architecture
Image Input Size: 224x224

CNN Layers: Conv2D → MaxPooling → Dropout → Dense Layers

Activation: ReLU & Softmax

Output: Multi-class classification (Tomato diseases)

🚀 Tech Stack
Python 🐍

TensorFlow / Keras 🤖

OpenCV 📷

Streamlit 🌐

Jupyter Notebook 📓

✅ Real-World Impact
Reduces dependency on manual disease identification

Helps farmers act early and save crops

Can be extended to other crops and integrated into mobile apps
