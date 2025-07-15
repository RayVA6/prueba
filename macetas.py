import streamlit as st
st.title("Subir y Mostrar una Imagen en Streamlit")

uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Para mostrar la imagen directamente
    st.image(uploaded_file, caption="Imagen subida.", use_column_width=True)

    # También puedes abrirla con PIL y luego mostrarla o procesarla
    try:
        image = Image.open(uploaded_file)
        st.write("Detalles de la imagen (PIL):")
        st.write(f"Formato: {image.format}")
        st.write(f"Tamaño: {image.size}")
        st.write(f"Modo: {image.mode}")

        # Si quieres hacer algún procesamiento con la imagen (opcional)
        # Por ejemplo, convertirla a escala de grises
        # gray_image = image.convert("L")
        # st.image(gray_image, caption="Imagen en escala de grises", use_column_width=True)

    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")

import cv2
import streamlit as st
import numpy as np
from PIL import Image
def brighten_image(image, amount):
img_bright = cv2.convertScaleAbs(image, beta=amount)
return img_bright
def blur_image(image, amount):
blur_img = cv2.GaussianBlur(image, (11, 11), amount)
return blur_img
def enhance_details(img):
hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
return hdr
def main_loop():
st.title("OpenCV Demo App")
st.subheader("This app allows you to play with Image filters!")
st.text("We use OpenCV and Streamlit for this demo")
blur_rate = st.sidebar.slider("Blurring", min_value=0.5, max_value=3.5)
brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)
apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')

image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
if not image_file:
    return None

original_image = Image.open(image_file)
original_image = np.array(original_image)

processed_image = blur_image(original_image, blur_rate)
processed_image = brighten_image(processed_image, brightness_amount)

if apply_enhancement_filter:
    processed_image = enhance_details(processed_image)

st.text("Original Image vs Processed Image")
st.image([original_image, processed_image])

if name == 'main':
main_loop()
