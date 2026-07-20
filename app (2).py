
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

st.title("Digit Recognition using CNN (MNIST)")
st.write("Upload an image of a digit (28x28 grayscale)")

model = tf.keras.models.load_model("mnist_cnn.h5")
uploaded_file = st.file_uploader("Choose an image...",type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = ImageOps.invert(image)                  # Invert to match MNIST format
    image = image.resize((28, 28))
    img_array = np.array(image).reshape(-1, 28, 28, 1) / 255.0

    st.image(image, caption="Uploaded Image", width=150)

    prediction = model.predict(img_array)
    st.success(f"Predicted Digit: {np.argmax(prediction)}")
