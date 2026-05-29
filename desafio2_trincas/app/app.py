"""
Interface Streamlit para deteccao de trincas em paredes.

Uso:
    streamlit run app/app.py
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path("models/best_crack_seg_yolo11n.pt")
DEFAULT_CONF = 0.25
DEFAULT_IOU  = 0.50


@st.cache_resource(show_spinner="Carregando modelo...")
def load_model(path: str) -> YOLO:
    return YOLO(path)


def run_inference(model: YOLO, image: np.ndarray,
                  conf: float, iou: float) -> tuple[np.ndarray, list[dict]]:
    results = model.predict(
        source       = image,
        conf         = conf,
        iou          = iou,
        imgsz        = 640,
        device       = "cpu",          # CPU para compatibilidade maxima no deploy
        verbose      = False,
        retina_masks = True,
    )[0]

    annotated = results.plot(line_width=2, font_size=12)
    # plot() retorna BGR (OpenCV) -- converte para RGB para Pillow/Streamlit
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    detections = []
    if results.boxes is not None:
        for i, box in enumerate(results.boxes):
            detections.append({
                "id":         i + 1,
                "confianca":  float(box.conf[0]),
                "classe":     model.names[int(box.cls[0])],
                "bbox_xywhn": box.xywhn[0].tolist(),
            })

    return annotated_rgb, detections


# ── Layout ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Detector de Trincas",
    page_icon=":hammer_and_wrench:",
    layout="wide",
)

st.title("Detector de Trincas e Fissuras em Paredes")
st.caption("YOLOv11n-seg treinado em imagens de inspecao de construcao civil.")

# Sidebar: parametros
with st.sidebar:
    st.header("Configuracoes")
    model_path_input = st.text_input("Caminho do modelo (.pt)", value=str(MODEL_PATH))
    conf_thresh = st.slider("Limiar de confianca", 0.05, 0.95, DEFAULT_CONF, 0.05,
                            help="Deteccoes abaixo deste valor sao descartadas.")
    iou_thresh  = st.slider("Limiar de IoU (NMS)", 0.10, 0.95, DEFAULT_IOU, 0.05,
                            help="Controla supressao de caixas sobrepostas.")
    st.divider()
    st.markdown("**Como usar:**\n1. Faca upload de uma imagem\n2. Ajuste os limiares\n"
                "3. Veja as trincas detectadas em vermelho")

# Verifica se modelo existe
if not Path(model_path_input).exists():
    st.error(f"Modelo nao encontrado: `{model_path_input}`\n\n"
             "Execute o treino primeiro: `python src/train.py`")
    st.stop()

model = load_model(model_path_input)

# Upload
uploaded = st.file_uploader(
    "Selecione uma imagem de parede",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded is not None:
    # Converte upload para numpy BGR (OpenCV)
    pil_img = Image.open(uploaded).convert("RGB")
    img_array = np.array(pil_img)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Imagem original")
        st.image(pil_img, use_container_width=True)

    with st.spinner("Detectando trincas..."):
        annotated, detections = run_inference(model, img_array, conf_thresh, iou_thresh)

    with col2:
        st.subheader(f"Resultado ({len(detections)} deteccao(oes))")
        st.image(annotated, use_container_width=True)

    # Tabela de deteccoes
    if detections:
        st.subheader("Detalhes das deteccoes")
        st.dataframe(
            [{"#": d["id"],
              "Classe": d["classe"],
              "Confianca": f"{d['confianca']:.2%}"}
             for d in detections],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Nenhuma trinca detectada com os limiares atuais. "
                "Tente reduzir o limiar de confianca.")

    # Download da imagem anotada
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        Image.fromarray(annotated).save(tmp.name, quality=95)
        with open(tmp.name, "rb") as f:
            st.download_button(
                label="Baixar imagem anotada",
                data=f,
                file_name=f"trincas_{uploaded.name}",
                mime="image/jpeg",
            )
else:
    st.info("Faca o upload de uma imagem para comecar.")
