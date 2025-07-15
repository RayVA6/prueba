import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
# No necesitas 'os' si el modelo está fijo en el repo

# --- Configuración de la Aplicación Streamlit ---
st.set_page_config(
    page_title="Detector de Objetos YOLO", # Título ajustado
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Detector de Objetos con YOLO") # Título ajustado
st.write("Sube una imagen para detectar objetos usando tu modelo pre-entrenado.")

# --- Carga del Modelo YOLO (Fijo en el Repositorio) ---
@st.cache_resource # Usa st.cache_resource para cargar el modelo una sola vez
def load_yolo_model():
    """
    Carga tu modelo pre-entrenado de YOLO.
    Asegúrate de que 'yolov11.pt' esté en la misma carpeta que tu script de Streamlit.
    """
    MODEL_PATH = 'best (floresuevasyabiertas).pt' # <--- ¡Aquí especificas la ruta de tu modelo!
    try:
        model = YOLO(MODEL_PATH)
        st.sidebar.success(f"Modelo '{MODEL_PATH}' cargado exitosamente.")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo YOLO: {e}")
        st.info(f"Asegúrate de que el archivo '{MODEL_PATH}' exista en el repositorio y sea un modelo YOLO válido.")
        return None

model = load_yolo_model()

if model is None:
    st.stop() # Detiene la ejecución si el modelo no se pudo cargar

# --- Interfaz para Subir Imagen y Realizar Detección ---
st.subheader("Subir Imagen para Detección")
uploaded_image_file = st.file_uploader(
    "Elige una imagen para detectar objetos",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_image_file is not None:
    st.subheader("Imagen Original")
    original_image = Image.open(uploaded_image_file)
    st.image(original_image, caption="Imagen subida", use_column_width=True)

    st.subheader("Resultados de la Detección")

    img_np = np.array(original_image)

    if img_np.shape[2] == 3:
        img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_np_bgr = img_np

    try:
        results = model.predict(source=img_np_bgr, conf=0.25, iou=0.7, show_labels=True, show_conf=True)

        for r in results:
            im_bgr = r.plot()
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            st.image(im_rgb, caption="Imagen con detecciones", use_column_width=True)

            if r.boxes:
                st.write(f"**Objetos detectados:**")
                for box in r.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    label = model.names[class_id]
                    st.write(f"- **{label}** (Confianza: {confidence:.2f})")
            else:
                st.write("No se detectaron objetos en esta imagen.")

    except Exception as e:
        st.error(f"Error durante la inferencia del modelo: {e}")
        st.info("Asegúrate de que el modelo cargado es compatible con las imágenes y la versión de ultralytics.")
else:
    st.info("Por favor, sube una imagen para realizar la detección.")

st.sidebar.header("Acerca de")
st.sidebar.info(
    "Esta aplicación usa un modelo YOLO pre-cargado desde el repositorio "
    "para realizar detección de objetos en imágenes subidas."
)
