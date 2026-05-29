"""
Interface Streamlit para deteccao de rachaduras em paredes.

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

# Modelos disponíveis para seleção no app
MODELOS = {
    "v3 — 1280px / 150 ep (recomendado)": "models/best_v3_nano_1280.pt",
    "v1 — 640px / 100 ep (baseline)":     "models/best_crack_seg_yolo11n.pt",
}

DEFAULT_CONF = 0.15   # 0.15 dá recall ~87% — melhor para inspeção em campo
DEFAULT_IOU  = 0.45


@st.cache_resource(show_spinner="Carregando modelo...")
def load_model(path: str) -> YOLO:
    return YOLO(path)


def run_inference(model: YOLO, image: np.ndarray,
                  conf: float, iou: float,
                  imgsz: int) -> tuple[np.ndarray, list[dict]]:
    results = model.predict(
        source       = image,
        conf         = conf,
        iou          = iou,
        imgsz        = imgsz,
        device       = "cpu",
        verbose      = False,
        retina_masks = True,
    )[0]

    annotated     = results.plot(line_width=2, font_size=12)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    detections = []
    if results.boxes is not None:
        for i, box in enumerate(results.boxes):
            x, y, w, h = box.xywhn[0].tolist()
            detections.append({
                "id":        i + 1,
                "confianca": float(box.conf[0]),
                "classe":    model.names[int(box.cls[0])],
                "x_centro":  round(x, 3),
                "y_centro":  round(y, 3),
                "largura":   round(w, 3),
                "altura":    round(h, 3),
            })

    return annotated_rgb, detections


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Detector de Rachaduras",
    page_icon="🔍",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configurações")

    modelo_label = st.selectbox("Modelo", list(MODELOS.keys()))
    model_path   = MODELOS[modelo_label]

    st.divider()

    conf_thresh = st.slider(
        "Limiar de confiança",
        min_value=0.05, max_value=0.95,
        value=DEFAULT_CONF, step=0.05,
        help="Valores menores detectam mais rachaduras mas geram mais falsos alarmes. "
             "0.15 recomendado para inspeção em campo.",
    )
    iou_thresh = st.slider(
        "Limiar de IoU (NMS)",
        min_value=0.10, max_value=0.95,
        value=DEFAULT_IOU, step=0.05,
        help="Controla a supressão de detecções sobrepostas.",
    )

    st.divider()
    st.markdown("""
**📋 Como usar:**
1. Faça upload ou tire uma foto
2. O modelo detecta as rachaduras automaticamente
3. Ajuste o limiar se necessário
4. Baixe a imagem anotada

**🎯 Referência de confiança:**
- `0.15` → mais sensível (menos falsos negativos)
- `0.25` → balanceado (padrão YOLO)
- `0.40` → mais preciso (menos falsos alarmes)
""")

    if not Path(model_path).exists():
        st.error(f"Modelo não encontrado:\n`{model_path}`")
        st.stop()

    imgsz = 1280 if "1280" in modelo_label else 640

# ── Cabeçalho ──────────────────────────────────────────────────────────────────
st.title("🔍 Detector de Rachaduras em Paredes")
st.caption(f"YOLOv11n-seg · {modelo_label} · conf ≥ {conf_thresh:.0%} · IoU {iou_thresh:.0%}")

model = load_model(model_path)

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📷 Envie uma foto da parede (JPG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Funciona com fotos tiradas no celular. "
         "Quanto maior a resolução, melhor a detecção de fissuras finas.",
)

# ── Inferência ─────────────────────────────────────────────────────────────────
if uploaded is not None:
    pil_img   = Image.open(uploaded).convert("RGB")
    img_array = np.array(pil_img)
    w_orig, h_orig = pil_img.size

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Imagem original")
        st.image(pil_img, use_container_width=True)
        st.caption(f"Resolução: {w_orig}×{h_orig} px")

    with st.spinner("🔎 Analisando rachaduras..."):
        annotated, detections = run_inference(
            model, img_array, conf_thresh, iou_thresh, imgsz
        )

    with col2:
        n = len(detections)
        if n == 0:
            st.subheader("✅ Nenhuma rachadura detectada")
        elif n == 1:
            st.subheader("⚠️ 1 rachadura detectada")
        else:
            st.subheader(f"🚨 {n} rachaduras detectadas")

        st.image(annotated, use_container_width=True)

    # ── Tabela de resultados ───────────────────────────────────────────────────
    if detections:
        st.subheader("📊 Detalhes das detecções")

        # Alerta visual baseado na quantidade
        if n >= 3:
            st.error(f"🚨 **{n} rachaduras** encontradas — recomenda-se inspeção técnica prioritária.")
        elif n >= 1:
            st.warning(f"⚠️ **{n} rachadura(s)** encontrada(s) — registrar e monitorar.")

        st.dataframe(
            [{
                "#":          d["id"],
                "Classe":     d["classe"],
                "Confiança":  f"{d['confianca']:.1%}",
                "X centro":   d["x_centro"],
                "Y centro":   d["y_centro"],
                "Largura":    d["largura"],
                "Altura":     d["altura"],
            } for d in detections],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("✅ Nenhuma rachadura detectada com os parâmetros atuais. "
                   "Reduza o limiar de confiança se suspeitar de falso negativo.")

    # ── Download ───────────────────────────────────────────────────────────────
    st.divider()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        Image.fromarray(annotated).save(tmp.name, quality=95)
        with open(tmp.name, "rb") as f:
            st.download_button(
                label="⬇️ Baixar imagem anotada",
                data=f,
                file_name=f"rachadura_{uploaded.name}",
                mime="image/jpeg",
                use_container_width=True,
            )

else:
    # ── Estado inicial ─────────────────────────────────────────────────────────
    st.info("👆 Faça o upload de uma foto para iniciar a análise.")

    st.markdown("""
### Como tirar boas fotos para o modelo
| ✅ Bom | ❌ Evitar |
|---|---|
| Luz natural ou iluminação uniforme | Flash direto (cria reflexos) |
| Câmera perpendicular à parede | Ângulo muito inclinado |
| Distância 50–150 cm da parede | Muito longe (perda de detalhe) |
| Foco nítido na superfície | Foto tremida ou desfocada |
| Parede seca | Parede molhada (reduz contraste) |
""")
