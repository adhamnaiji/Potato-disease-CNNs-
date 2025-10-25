🥔 Potato Disease Classification using CNNs
A deep learning solution for automated detection and classification of potato plant diseases using Convolutional Neural Networks. This project aims to help farmers identify diseases in potato plants early, enabling timely treatment and reducing crop losses.

📋 Table of Contents
About the Project

Disease Categories

Model Architecture

Installation

Usage

Dataset

Results

Project Structure

Contributing


🎯 About the Project
Potato diseases cause significant agricultural losses worldwide. Manual inspection is time-consuming and requires expertise. This project leverages computer vision and deep learning to automatically classify potato leaf diseases from images, providing:

Fast diagnosis - Instant disease detection from leaf images

High accuracy - CNN-based classification with robust performance

Easy deployment - Ready-to-use Jupyter notebooks

Scalable solution - Can be extended to other plant diseases

🌱 Disease Categories
The model classifies potato leaves into the following categories:

Healthy - No disease detected

Early Blight - Caused by Alternaria solani

Late Blight - Caused by Phytophthora infestans

🏗️ Model Architecture
The CNN architecture consists of:

Input Layer - Accepts RGB images of potato leaves

Convolutional Layers - Extract spatial features from images

Pooling Layers - Reduce dimensionality while preserving important features

Dropout Layers - Prevent overfitting during training

Dense Layers - Final classification

Output Layer - 3-class softmax activation

💻 Installation
Prerequisites
Python 3.7 or higher

pip package manager

Git

Setup Instructions
Clone the repository

bash
git clone https://github.com/adhamnaiji/Potato-disease-CNNs-.git
cd Potato-disease-CNNs-
Create a virtual environment (recommended)

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

bash
pip install tensorflow numpy pandas matplotlib opencv-python jupyter scikit-learn pillow
🚀 Usage
Running the Jupyter Notebook
Start Jupyter Notebook:

bash
jupyter notebook
Open the main notebook file in your browser

Run all cells sequentially to:

Load and preprocess the dataset

Train the CNN model

Evaluate model performance

Make predictions on new images

Making Predictions
To classify a new potato leaf image:

python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = load_model('potato_disease_model.h5')

# Load and preprocess image
img_path = 'path/to/potato_leaf.jpg'
img = image.load_img(img_path, target_size=(256, 256))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
prediction = model.predict(img_array)
classes = ['Early Blight', 'Healthy', 'Late Blight']
result = classes[np.argmax(prediction)]
print(f"Predicted: {result}")
📊 Dataset
The dataset contains images of potato leaves categorized by disease type. Images are preprocessed with:

Resizing to consistent dimensions

Normalization (pixel values 0-1)

Data augmentation for training robustness

Train/validation/test split

Dataset source: PlantVillage or similar agricultural disease datasets

📈 Results
Model Performance
Metric	Score
Training Accuracy	~XX%
Validation Accuracy	~XX%
Test Accuracy	~XX%
F1-Score	~XX%
Replace XX% with your actual model performance metrics

Sample Predictions
Add screenshots of your model's predictions here

Training History
Include loss and accuracy plots showing model training progress

📁 Project Structure
text
Potato-disease-CNNs-/
│
├── notebooks/                 # Jupyter notebooks
│   └── training.ipynb        # Main training notebook
│
├── data/                     # Dataset directory
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/                   # Saved model files
│   └── potato_disease_model.h5
│
├── images/                   # Sample images and results
│
├── requirements.txt          # Python dependencies
├── LICENSE                   # Project license
└── README.md                # This file
🔮 Future Improvements
 Deploy as web application using Flask/FastAPI

 Create mobile app for farmers

 Expand to detect more potato diseases

 Implement transfer learning (ResNet, EfficientNet)

 Add disease severity estimation

 Real-time detection using webcam/smartphone

 Multi-language support for global accessibility

 Integration with IoT sensors for automated monitoring

🤝 Contributing
Contributions make the open-source community an amazing place to learn and create. Any contributions you make are greatly appreciated.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request
