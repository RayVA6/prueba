import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import pandas as pd
import io # Para manejar la descarga del CSV

# --- Configuración de la Aplicación Streamlit ---
st.set_page_config(
    page_title="Detector de Objetos YOLO Múltiple + CSV",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Detector de Objetos con YOLO (Múltiples Imágenes y Exportación CSV)")
st.write("Sube una o varias imágenes para detectar objetos y exportar los resultados.")

# --- Carga del Modelo YOLO (Fijo en el Repositorio de GitHub) ---
@st.cache_resource
def load_yolo_model():
    MODEL_PATH = 'best (floresuevasyabiertas).pt' # <--- ¡IMPORTANTE! Reemplaza esto con el nombre exacto de tu archivo de modelo
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

# --- Interfaz para Subir Múltiples Imágenes ---
st.subheader("Subir Imágenes para Detección")
uploaded_image_files = st.file_uploader(
    "Elige una o más imágenes para detectar objetos",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True # <--- ¡CAMBIO CLAVE AQUÍ! Permite múltiples archivos
)

# Lista para almacenar los resultados de todas las imágenes
all_detections_summary = []

if uploaded_image_files: # Si hay archivos subidos
    st.write(f"Procesando {len(uploaded_image_files)} imágenes...")

    # Crear columnas para mostrar las imágenes en un diseño de cuadrícula
    cols = st.columns(3) # Ajusta el número de columnas según prefieras (ej. 2, 3, 4)
    col_idx = 0

    for i, uploaded_file in enumerate(uploaded_image_files):
        # Mostrar el nombre del archivo
        st.info(f"Procesando: {uploaded_file.name}")

        # Abrir y procesar la imagen
        original_image = Image.open(uploaded_file)
        img_np = np.array(original_image)

        if img_np.shape[2] == 3:
            img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_np_bgr = img_np

        try:
            results = model.predict(source=img_np_bgr, conf=0.25, iou=0.7, show_labels=True, show_conf=True)

            for r in results: # Solo habrá un 'r' por imagen en este caso
                im_bgr = r.plot()
                im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

                # --- Conteo de Detecciones por Clase (como ya lo tenías) ---
                class_counts = {}
                for box in r.boxes:
                    class_id = int(box.cls)
                    label = model.names[class_id]
                    class_counts[label] = class_counts.get(label, 0) + 1

                num_total_detections = len(r.boxes)

                # --- Almacenar resultados para el CSV ---
                # Crear una entrada para esta imagen
                image_detection_data = {"Imagen": uploaded_file.name}
                if num_total_detections == 0:
                    image_detection_data["Total Detecciones"] = 0
                    image_detection_data["Estado"] = "No se detectaron objetos."
                else:
                    image_detection_data["Total Detecciones"] = num_total_detections
                    image_detection_data["Estado"] = "Objetos detectados."
                    for label, count in class_counts.items():
                        image_detection_data[label] = count # Añade una columna por cada clase

                all_detections_summary.append(image_detection_data)


                # --- Mostrar resultados en la interfaz ---
                with cols[col_idx]:
                    st.image(im_rgb, caption=f"Detecciones en {uploaded_file.name}", use_column_width=True)
                    
                    # Mensaje de resumen bajo la imagen
                    summary_message = f"Total: {num_total_detections} objetos."
                    if num_total_detections > 0:
                        summary_message += " Desglose:"
                        for label, count in class_counts.items():
                            summary_message += f" {label}: {count}"
                    st.markdown(f"**{summary_message}**") # Usamos Markdown para negritas


                col_idx = (col_idx + 1) % len(cols) # Mover a la siguiente columna

        except Exception as e:
            st.error(f"Error procesando '{uploaded_file.name}': {e}")
            all_detections_summary.append({"Imagen": uploaded_file.name, "Total Detecciones": "ERROR", "Estado": f"Error: {e}"})

    # --- Mostrar Tabla de Resumen y Botón de Descarga CSV ---
    if all_detections_summary:
        st.subheader("Resumen de Todas las Detecciones")
        # Convertir la lista de diccionarios a un DataFrame de Pandas
        df_summary = pd.DataFrame(all_detections_summary)
        # Rellenar valores NaN (para clases que no aparecieron en todas las imágenes) con 0
        df_summary = df_summary.fillna(0)
        
        # Mover las columnas de conteo al principio, si no están ya
        # Primero, obtén todas las columnas de clase que pueden existir
        class_columns = [col for col in df_summary.columns if col not in ["Imagen", "Total Detecciones", "Estado"]]
        # Ordena las columnas
        ordered_columns = ["Imagen", "Total Detecciones", "Estado"] + sorted(class_columns)
        df_summary = df_summary[ordered_columns]


        st.dataframe(df_summary) # Mostrar el DataFrame en Streamlit

        # Crear un botón para descargar el CSV
        csv_buffer = io.StringIO()
        df_summary.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Descargar Resultados como CSV",
            data=csv_buffer.getvalue(),
            file_name="resultados_deteccion.csv",
            mime="text/csv",
            help="Haz clic para descargar un archivo CSV con el conteo de detecciones por imagen."
        )

else:
    st.info("Por favor, sube una o más imágenes para realizar la detección de objetos")

st.sidebar.header("Acerca de")
st.sidebar.info(
    "Esta aplicación permite subir múltiples imágenes, detectar objetos con un modelo YOLO "
    "pre-entrenado y exportar un resumen de las detecciones a un archivo CSV."
)
