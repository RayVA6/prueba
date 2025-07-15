import streamlit as st
from PIL import Image
import numpy as np
import cv2 # Necesario para algunas operaciones de imagen si ultralytics lo usa internamente o para procesar la salida
from ultralytics import YOLO # Importa la clase YOLO de ultralytics

# --- Configuración de la Aplicación Streamlit ---
st.set_page_config(
    page_title="Detector de Objetos YOLOv11",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Detector de Objetos con YOLOv11")
st.write("Sube una imagen para detectar objetos usando tu modelo pre-entrenado.")

# --- Carga del Modelo YOLOv11 ---
@st.cache_resource # Usa st.cache_resource para cargar el modelo una sola vez
def load_yolo_model():
    """
    Carga tu modelo pre-entrenado de YOLOv11.
    Asegúrate de que 'yolov11.pt' esté en la misma carpeta que tu script de Streamlit
    o proporciona la ruta completa a tu archivo de modelo.
    """
    try:
        # Reemplaza 'yolov11.pt' con el nombre de tu archivo de modelo
        model = YOLO('yolov11.pt')
        st.success("Modelo YOLOv11 cargado exitosamente.")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo YOLOv11: {e}")
        st.info("Asegúrate de que el archivo 'yolov11.pt' exista y esté en la ruta correcta.")
        return None

model = load_yolo_model()

if model is None:
    st.stop() # Detiene la ejecución si el modelo no se pudo cargar

# --- Interfaz para Subir Imagen ---
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    # Mostrar la imagen original
    st.subheader("Imagen Original")
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="Imagen subida", use_column_width=True)

    st.subheader("Resultados de la Detección")
    
    # Convertir la imagen de PIL a un formato que YOLO pueda procesar (ej. NumPy array)
    # ultralytics puede trabajar directamente con PIL.Image, pero a veces es útil convertir a numpy
    img_np = np.array(original_image)
    
    # Convertir de RGB a BGR si la imagen es RGB (PIL es RGB, OpenCV es BGR)
    # ultralytics generalmente maneja esto, pero es una buena práctica si hay problemas
    if img_np.shape[2] == 3: # Si es una imagen a color
        img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_np_bgr = img_np # Si es escala de grises o ya BGR

    # Realizar la inferencia
    try:
        # La función 'predict' de YOLO puede tomar una imagen PIL, un array NumPy o una ruta de archivo.
        # Esto devuelve una lista de objetos 'Results'
        results = model.predict(source=img_np_bgr, conf=0.25, iou=0.7, show_labels=True, show_conf=True)
        
        # Procesar y mostrar los resultados
        for r in results:
            # r.plot() devuelve una imagen con las detecciones dibujadas (NumPy array en BGR)
            im_bgr = r.plot()
            # Convertir de BGR a RGB para mostrar correctamente en Streamlit (st.image espera RGB)
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            st.image(im_rgb, caption="Imagen con detecciones", use_column_width=True)

            # Opcional: Mostrar detalles de las detecciones
            if r.boxes: # Si hay cajas detectadas
                st.write(f"**Objetos detectados:**")
                for box in r.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    label = model.names[class_id] # Obtener el nombre de la clase
                    st.write(f"- **{label}** (Confianza: {confidence:.2f})")
            else:
                st.write("No se detectaron objetos en esta imagen.")

    except Exception as e:
        st.error(f"Error durante la inferencia del modelo: {e}")
        st.info("Asegúrate de que tu modelo YOLOv11 esté correctamente entrenado y sea compatible con la versión de ultralytics.")

st.sidebar.header("Acerca de")
st.sidebar.info(
    "Esta aplicación demuestra la detección de objetos usando un modelo YOLOv11 pre-entrenado "
    "y la librería Streamlit para la interfaz de usuario."
)

