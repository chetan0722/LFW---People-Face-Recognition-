
import streamlit as st
import cv2
import joblib
import numpy as np
from PIL import Image

model = joblib.load("knn_face_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

st.title("Face Recognition using KNN")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    image = np.array(image)

    st.image(image)

    image = cv2.resize(image,(100,100))
    image = image.flatten().reshape(1,-1)

    image = scaler.transform(image)
    image = pca.transform(image)

    prediction = model.predict(image)

    st.success(prediction[0])
