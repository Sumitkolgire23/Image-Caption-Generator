import streamlit as st
from src.inference import generate_caption
from PIL import Image

st.title('Image Caption Generator — Flickr8k demo')
uploaded = st.file_uploader('Upload an image', type=['jpg','jpeg','png'])
if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    with open('temp.jpg','wb') as f:
        f.write(uploaded.getbuffer())
    if st.button('Generate caption'):
        caption = generate_caption('temp.jpg')
        st.write('**Caption:**', caption)
