# LFW People Face Recognition using KNN & PCA

## Project Overview

This project performs **Face Recognition** using the **Labeled Faces in the Wild (LFW)** dataset. The model uses **Principal Component Analysis (PCA)** for dimensionality reduction and **K-Nearest Neighbors (KNN)** for classification.

The application is deployed using **Streamlit**, allowing users to upload a face image and receive the predicted person's name.

---

## Features

* Face Recognition using Machine Learning
* PCA for feature extraction
* KNN Classifier
* Image preprocessing with OpenCV
* Streamlit Web Application
* Model serialization using Joblib

---

## Technologies Used

* Python
* NumPy
* OpenCV
* Scikit-learn
* Streamlit
* Pillow
* Joblib

---

## Dataset

**Labeled Faces in the Wild (LFW)**

The dataset contains face images of well-known personalities captured under different lighting conditions, poses, and facial expressions.

---

## Project Structure

```text
LFW-People-Face-Recognition/
│
├── app.py
├── requirements.txt
├── knn_face_model.pkl
├── scaler.pkl
├── pca.pkl
└── README.md
```

---

## Machine Learning Pipeline

```text
Load Dataset
      ↓
Image Preprocessing
      ↓
Resize Images
      ↓
Flatten Images
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
PCA
      ↓
KNN Classifier
      ↓
Prediction
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/chetan0722/LFW---People-Face-Recognition-.git
```

Move into the project directory

```bash
cd LFW---People-Face-Recognition-
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## Model Performance

* Algorithm: K-Nearest Neighbors (KNN)
* Feature Extraction: PCA
* Dataset: LFW Faces
* Accuracy: ~28% on the selected multi-class subset

> Note: The LFW dataset is a challenging real-world face recognition benchmark with variations in pose, lighting, and facial expressions. Using raw pixel values with PCA and KNN provides a baseline solution. Higher accuracy can be achieved using deep learning-based face embeddings such as FaceNet or ArcFace.

---

## Future Improvements

* Face Detection before Recognition
* Face Alignment
* FaceNet Embeddings
* ArcFace Integration
* Webcam Support
* Real-time Face Recognition
* Model Deployment on Streamlit Cloud


