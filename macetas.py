import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# --- Configuración de la Aplicación Streamlit ---
st.set_page_config(
    page_title="Detector de Objetos YOLO",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Detector de Objetos con YOLO")
st.write("Sube una imagen para detectar objetos usando tu modelo pre-entrenado.")

# --- Carga del Modelo YOLO (Fijo en el Repositorio de GitHub) ---
@st.cache_resource
def load_yolo_model():
    MODEL_PATH = 'best (floresnuevasyabiertas).pt' # <--- ¡IMPORTANTE! Reemplaza esto con el nombre exacto de tu archivo de modelo
    try:
        model = YOLO(MODEL_PATH)
        st.sidebar.success(f"Modelo '{MODEL_PATH}' cargado exitosamente.")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo YOLO: {e}")
        st.info(f"Asegúrate de que el archivo '{MODEL_PATH}' exista en la misma carpeta del script en tu repositorio de GitHub y sea un modelo YOLO válido.")
        return None

model = load_yolo_model()

if model is None:
    st.stop()

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

            # --- MODIFICACIÓN CLAVE AQUÍ ---
            # 1. Contar detecciones por clase
            class_counts = {}
            for box in r.boxes:
                class_id = int(box.cls)
                label = model.names[class_id]
                class_counts[label] = class_counts.get(label, 0) + 1

            num_total_detections = len(r.boxes)

            # 2. Construir el mensaje de resumen
            summary_message = f"Se detectaron {num_total_detections} objetos en total."

            if num_total_detections > 0:
                summary_message += "\n" # Nueva línea para el desglose
                for label, count in class_counts.items():
                    summary_message += f"- {label}: {count} {'detección' if count == 1 else 'detecciones'}\n"

            # 3. Mostrar la imagen con el resumen en el caption
            st.image(im_rgb, caption=f"Imagen con detecciones.\n{summary_message}", use_column_width=True)

            # --- Ya no es necesario el bloque de "Detalles de las detecciones" si el resumen es suficiente ---
            # Si aún quieres el listado detallado, puedes mantener este bloque:
            if r.boxes:
                st.write(f"**Detalles individuales de las detecciones:**")
                for box in r.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    label = model.names[class_id]
                    st.write(f"- **{label}** (Confianza: {confidence:.2f})")
            # else:
            #     st.write("No se detectaron objetos en esta imagen.")


    except Exception as e:
        st.error(f"Error durante la inferencia del modelo: {e}")
        st.info("Asegúrate de que el modelo es compatible con la imagen y la versión de ultralytics.")
else:
    st.info("Por favor, sube una imagen para realizar la detección de objetos.")

st.sidebar.header("Acerca de")
st.sidebar.info(
    "Esta aplicación demuestra la detección de objetos usando un modelo YOLO "
    "pre-entrenado cargado desde el repositorio de GitHub."
)
