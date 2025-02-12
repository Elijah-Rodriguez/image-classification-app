import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.title("Location Image Classifier")
st.text("Provide URL of Location Image for image classification")

@st.cache_resource
def load_model():
  model = tf.keras.models.load_model('./model.h5')
  return model

with st.spinner('Loading Model....'):
  model = load_model()

classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def decode_img(image):
  img = tf.image.decode_jpeg(image, channels=3)  
  img = tf.image.resize(img,[150,150])
  return np.expand_dims(img, axis=0)

path = st.text_input('Enter Image URL to Classify.. ', 'https://www.nps.gov/glac/learn/nature/images/Folds.jpg')

if path is not None:
  try:
    content = requests.get(path).content
  except (NameError, requests.exceptions.MissingSchema):
    pass
  
  st.write("Predicted Class :")
  with st.spinner('classifying.....'):
    label =np.argmax(model.predict(decode_img(content)),axis=1)
    st.write(classes[label[0]])    
  st.write("")
  image = Image.open(BytesIO(content))
  st.image(image, caption='Classified Image', use_container_width=True)