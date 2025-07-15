import streamlit as st
from PIL import Image
import numpy as np
import cv2 # Asegúrate de que opencv-python esté en requirements.txt
from ultralytics import YOLO # Asegúrate de que ultralytics esté en requirements.txt

# --- Configuración de la Aplicación Streamlit ---
st.set_page_config(
    page_title="Detector de Objetos YOLO", # Título de la pestaña del navegador
    page_icon="🔍",                       # Icono en la pestaña
    layout="wide",                       # Distribución ancha de la página
    initial_sidebar_state="expanded"     # Barra lateral expandida por defecto
)

st.title("Detector de Objetos con YOLO")
st.write("Sube una imagen para detectar objetos usando tu modelo pre-entrenado.")

# --- Carga del Modelo YOLO (Fijo en el Repositorio de GitHub) ---
# Usamos st.cache_resource para cargar el modelo una sola vez,
# lo que es crucial para la eficiencia en Streamlit.
@st.cache_resource
def load_yolo_model():
    """
    Carga tu modelo pre-entrenado de YOLO.
    Asegúrate de que 'yolov11.pt' esté en la misma carpeta que tu script de Streamlit
    en tu repositorio de GitHub.
    """
    MODEL_PATH = 'best (floresuevasyabiertas).pt' # <--- ¡IMPORTANTE! Reemplaza esto con el nombre exacto de tu archivo de modelo
    try:
        model = YOLO(MODEL_PATH)
        st.sidebar.success(f"Modelo '{MODEL_PATH}' cargado exitosamente.")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo YOLO: {e}")
        st.info(f"Asegúrate de que el archivo '{MODEL_PATH}' exista en la misma carpeta del script en tu repositorio de GitHub y sea un modelo YOLO válido.")
        return None

# Intentar cargar el modelo al inicio de la aplicación
model = load_yolo_model()

# Si el modelo no pudo cargarse, detiene la ejecución del resto de la app
if model is None:
    st.stop()

# --- Interfaz para Subir Imagen y Realizar Detección ---
st.subheader("Subir Imagen para Detección")
uploaded_image_file = st.file_uploader(
    "Elige una imagen para detectar objetos",
    type=["jpg", "jpeg", "png", "bmp", "webp"] # Tipos de archivo de imagen permitidos
)

if uploaded_image_file is not None:
    st.subheader("Imagen Original")
    original_image = Image.open(uploaded_image_file)
    st.image(original_image, caption="Imagen subida", use_column_width=True)

    st.subheader("Resultados de la Detección")
    
    # Convertir la imagen de PIL a un array NumPy para procesamiento
    img_np = np.array(original_image)
    
    # ultralytics/OpenCV esperan imágenes en formato BGR. PIL lee en RGB.
    # Esta conversión es necesaria para asegurar que los colores se muestren correctamente
    # y que el modelo interprete la imagen como espera.
    if img_np.shape[2] == 3: # Solo si la imagen tiene 3 canales (color)
        img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_np_bgr = img_np # Si es escala de grises o ya BGR (poco probable de PIL)

    # Realizar la inferencia con el modelo YOLO
    try:
        # model.predict() devuelve los resultados de la detección
        results = model.predict(source=img_np_bgr, conf=0.25, iou=0.7, show_labels=True, show_conf=True)
        
        # Iterar sobre los resultados (puede haber múltiples si se procesan varias imágenes, aunque aquí solo una)
        for r in results:
            # r.plot() dibuja las cajas delimitadoras, etiquetas y confianzas en la imagen
            im_bgr = r.plot()
            # Convertir de BGR de nuevo a RGB para que st.image la muestre correctamente
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            st.image(im_rgb, caption="Imagen con detecciones", use_column_width=True)

            # Opcional: Mostrar una lista textual de los objetos detectados
            if r.boxes: # Si se detectaron cajas
                st.write(f"**Objetos detectados:**")
                for box in r.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    # model.names mapea el ID de la clase a su nombre
                    label = model.names[class_id]
                    st.write(f"- **{label}** (Confianza: {confidence:.2f})")
            else:
                st.write("No se detectaron objetos en esta imagen.")

    except Exception as e:
        st.error(f"Error durante la inferencia del modelo: {e}")
        st.info("Asegúrate de que el modelo es compatible con la imagen y la versión de ultralytics.")
else:
    st.info("Por favor, sube una imagen para realizar la detección de objetos.")

# --- Información en la Barra Lateral ---
st.sidebar.header("Acerca de")
st.sidebar.info(
    "Esta aplicación demuestra la detección de objetos usando un modelo YOLO "
    "pre-entrenado cargado desde el repositorio de GitHub."
)
