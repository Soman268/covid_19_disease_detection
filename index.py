import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np

classifier = load_model('my_1model.h5')

st.title("Model for Covid-19 Disease Detection using x-ray image classification!")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    st.image(uploaded_file)
    # for only one prediction
    img = Image.open(uploaded_file)
    img_array = np.array(img)
    cv2.imwrite('temp.jpg', img_array)
    test_image = image.load_img('temp.jpg', target_size=(64, 64, 3))
    # img = image.resize((64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    if result[0][0] == 1:
        prediction = 'Normal'
    else:
        prediction = 'Covid'
    st.title(prediction)