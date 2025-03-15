import streamlit as st
from PIL import Image
from utils import load_and_preprocess_image, predict_image_class

st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader('Upload an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            prediction = predict_image_class(uploaded_image)
            st.success(f'Prediction: {str(prediction)}')
