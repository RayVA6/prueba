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


