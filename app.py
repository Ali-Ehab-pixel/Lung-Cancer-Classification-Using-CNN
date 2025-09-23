
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

def load_my_model():
  return load_model()

model = load_my_model()

img_height, img_width = 128,128
class_labels = ['adenocarcinoma','squamous_cell_carcinoma','benign']

st.title("Lung Cancer Classification")
st.write("Upload an image to classify its category.")

uploaded_file = st.file_uploader("Choose an Image...", type=['jpg','jpg','png'])

if uploaded_file is not None:

  image = Image.open(uploaded_file)
  st.image(image, caption='Uploaded Image', use_column_width=True)

  img = image.resize(img_height, img_width)
  img_array = img_to_array(img) / 255.0
  img_array = np.expand_dims(img_array, axis=0)

  predictions = model.predict(img_array)
  predicted_class = class_labels[np.argmax(predictions)]
  confidence = np.max(predictions) * 100

  st.write(f"Predicted Category: {predicted_class}")
  st.write(f"Confidence: {confidence:.2f}%")