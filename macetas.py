import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import os # Para manejar rutas de archivos temporales

# --- Configuración de la Aplicación Streamlit ---
st.set_page_config(
    page_title="Detector de Objetos YOLO Dinámico",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Detector de Objetos con YOLO (Carga de Modelo Dinámica)")
st.write("Sube tu archivo de modelo YOLO (.pt) y luego una imagen para la detección.")

# --- Interfaz para Subir el Archivo del Modelo ---
st.sidebar.header("Cargar Modelo YOLO")
uploaded_model_file = st.sidebar.file_uploader(
    "Elige tu archivo de modelo YOLO (.pt)",
    type=["pt"],
    help="Sube tu archivo de modelo YOLO (ej. yolov11.pt, yolov8n.pt)"
)

model = None
temp_model_path = None

if uploaded_model_file is not None:
    # Guardar el archivo subido temporalmente para que YOLO pueda cargarlo
    # Creamos un directorio temporal si no existe
    if not os.path.exists("temp_models"):
        os.makedirs("temp_models")
        
    temp_model_path = os.path.join("temp_models", uploaded_model_file.name)
    with open(temp_model_path, "wb") as f:
        f.write(uploaded_model_file.getbuffer())
    
    try:
        # Cargar el modelo desde la ruta temporal
        model = YOLO(temp_model_path)
        st.sidebar.success(f"Modelo '{uploaded_model_file.name}' cargado exitosamente.")
    except Exception as e:
        st.sidebar.error(f"Error al cargar el modelo: {e}")
        st.sidebar.info("Asegúrate de que el archivo es un modelo YOLO válido y compatible con la versión de ultralytics.")
        model = None
else:
    st.sidebar.info("Por favor, sube un archivo de modelo YOLO (.pt) para comenzar.")


# --- Interfaz para Subir Imagen y Realizar Detección ---
if model is not None:
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
        
        if img_np.shape[2] == 3: # Si es una imagen a color
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
else:
    st.info("Sube tu archivo de modelo YOLO (.pt) en la barra lateral para activar la detección de objetos.")

st.sidebar.header("Acerca de")
st.sidebar.info(
    "Esta aplicación permite cargar dinámicamente un modelo YOLO pre-entrenado (.pt) "
    "y realizar detección de objetos en imágenes subidas."
)
