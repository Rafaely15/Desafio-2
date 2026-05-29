"""
Inspetor de Qualidade — Detecção de Rachaduras em Paredes
Interface profissional para inspeção em canteiro de obras.

Uso:
    streamlit run app/app.py
"""

import base64
import io
import json
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ── Constantes ─────────────────────────────────────────────────────────────────
MODELOS = {
    "v3 — 1280px (recomendado)": "models/best_v3_nano_1280.pt",
    "v1 — 640px  (baseline)":    "models/best_crack_seg_yolo11n.pt",
}
HISTORICO_FILE = Path("results/historico_inspecoes.json")
HISTORICO_FILE.parent.mkdir(exist_ok=True)

# ── CSS personalizado ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Inspetor de Qualidade — Engenharia Civil",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Remove padding padrão do Streamlit */
.block-container { padding-top: 0rem !important; }
[data-testid="stSidebar"] > div:first-child { padding-top: 0rem; }

/* Cabeçalho principal */
.header-bar {
    background: linear-gradient(90deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    color: white;
    padding: 14px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 3px solid #e94560;
    margin: -1rem -1rem 1rem -1rem;
    border-radius: 0;
}
.header-title {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 2px;
    color: white;
    text-transform: uppercase;
}
.header-icons { font-size: 1.4rem; }

/* Logo sidebar */
.logo-box {
    background: #1a1a2e;
    color: white;
    text-align: center;
    padding: 18px 8px 12px 8px;
    border-radius: 10px;
    margin-bottom: 12px;
    border: 1px solid #333;
}
.logo-box .logo-icon { font-size: 2.5rem; }
.logo-box .logo-text {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #aab;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Títulos de seção */
.section-title {
    background: #1a1a2e;
    color: #aab;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 6px 10px;
    border-radius: 6px;
    margin: 6px 0 8px 0;
}

/* Card do histórico */
.hist-card {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 7px 8px;
    margin-bottom: 6px;
}
.hist-card.alert  { border-left: 4px solid #e94560; }
.hist-card.ok     { border-left: 4px solid #28a745; }
.hist-thumb { width: 52px; height: 38px; object-fit: cover; border-radius: 4px; }
.hist-info  { flex: 1; }
.hist-name  { font-size: 0.73rem; font-weight: 600; color: #222; }
.hist-date  { font-size: 0.67rem; color: #888; }
.hist-badge-alert { color: #e94560; font-size: 1rem; }
.hist-badge-ok    { color: #28a745; font-size: 1rem; }

/* Área de resultado */
.status-banner-ok {
    background: linear-gradient(90deg, #1a1a2e, #0f3460);
    color: white;
    text-align: center;
    padding: 8px;
    border-radius: 8px 8px 0 0;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.status-banner-alert {
    background: linear-gradient(90deg, #7b1e1e, #e94560);
    color: white;
    text-align: center;
    padding: 8px;
    border-radius: 8px 8px 0 0;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.detect-badge-alert {
    background: linear-gradient(90deg, #7b1e1e, #e94560);
    color: white;
    text-align: center;
    padding: 8px;
    border-radius: 0 0 8px 8px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.detect-badge-ok {
    background: linear-gradient(90deg, #1a4d2e, #28a745);
    color: white;
    text-align: center;
    padding: 8px;
    border-radius: 0 0 8px 8px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.img-frame {
    border: 2px solid #1a1a2e;
    border-top: none;
    border-bottom: none;
}

/* Botões de ação */
div[data-testid="column"] .stButton > button {
    width: 100%;
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    border-radius: 6px;
    padding: 10px 6px;
    border: none;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando modelo...")
def load_model(path: str) -> YOLO:
    return YOLO(path)


def pil_to_b64(img: Image.Image, size=(60, 44)) -> str:
    img_small = img.copy()
    img_small.thumbnail(size)
    buf = io.BytesIO()
    img_small.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def run_inference(model: YOLO, image: np.ndarray,
                  conf: float, iou: float, imgsz: int):
    results = model.predict(
        source=image, conf=conf, iou=iou,
        imgsz=imgsz, device="cpu", verbose=False, retina_masks=True,
    )[0]
    annotated_rgb = cv2.cvtColor(results.plot(line_width=3), cv2.COLOR_BGR2RGB)
    detections = []
    if results.boxes is not None:
        for i, box in enumerate(results.boxes):
            detections.append({
                "id": i + 1,
                "confianca": float(box.conf[0]),
                "classe": model.names[int(box.cls[0])],
            })
    return annotated_rgb, detections


def load_historico() -> list:
    if HISTORICO_FILE.exists():
        try:
            return json.loads(HISTORICO_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_historico(hist: list):
    HISTORICO_FILE.write_text(
        json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── Session state ──────────────────────────────────────────────────────────────
if "historico" not in st.session_state:
    st.session_state.historico = load_historico()
if "resultado" not in st.session_state:
    st.session_state.resultado = None
if "obs_texto" not in st.session_state:
    st.session_state.obs_texto = ""
if "local_input" not in st.session_state:
    st.session_state.local_input = ""


# ── Cabeçalho ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <div class="header-icons">☰</div>
  <div class="header-title">🏗️ &nbsp; Inspetor de Qualidade — Engenharia Civil</div>
  <div class="header-icons">👷</div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-box">
      <div class="logo-icon">🏗️</div>
      <div class="logo-text">Residência IA</div>
    </div>
    """, unsafe_allow_html=True)

    # Configurações colapsáveis
    with st.expander("⚙️ Configurações", expanded=False):
        modelo_label = st.selectbox("Modelo", list(MODELOS.keys()), label_visibility="collapsed")
        conf_thresh  = st.slider("Confiança", 0.05, 0.95, 0.15, 0.05)
        iou_thresh   = st.slider("IoU (NMS)", 0.10, 0.95, 0.45, 0.05)

    st.markdown('<div class="section-title">📋 Diário de Obra (Histórico)</div>',
                unsafe_allow_html=True)

    hist = st.session_state.historico
    if not hist:
        st.caption("Nenhuma inspeção salva ainda.")
    else:
        for item in reversed(hist[-10:]):   # mostra os 10 mais recentes
            alerta = item["n_det"] > 0
            cls    = "alert" if alerta else "ok"
            badge  = "❗" if alerta else "✅"
            thumb  = item.get("thumb", "")
            thumb_html = f'<img class="hist-thumb" src="data:image/jpeg;base64,{thumb}">' if thumb else "📷"
            status_txt = f"{item['n_det']} rachadura(s)" if alerta else "Sem falhas"
            st.markdown(f"""
            <div class="hist-card {cls}">
              {thumb_html}
              <div class="hist-info">
                <div class="hist-name">{item['local'][:22]}</div>
                <div class="hist-date">{item['data']} · {status_txt}</div>
              </div>
              <span class="hist-badge-{'alert' if alerta else 'ok'}">{badge}</span>
            </div>
            """, unsafe_allow_html=True)

    if hist:
        if st.button("🗑️ Limpar histórico", use_container_width=True):
            st.session_state.historico = []
            save_historico([])
            st.rerun()


# ── Conteúdo principal ─────────────────────────────────────────────────────────
model_path = MODELOS[modelo_label]
if not Path(model_path).exists():
    st.error(f"Modelo não encontrado: `{model_path}`. Execute o treino primeiro.")
    st.stop()

model = load_model(model_path)
imgsz = 1280 if "1280" in modelo_label else 640

# Entrada de dados: upload OU câmera
col_up, col_cam = st.columns(2)

with col_up:
    st.markdown("#### 📁 Enviar foto")
    uploaded = st.file_uploader(
        "Arraste ou clique para selecionar",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

with col_cam:
    st.markdown("#### 📷 Tirar foto")
    camera_img = st.camera_input(
        "Clique em 'Take Photo'",
        label_visibility="collapsed",
    )

# Local da inspeção
local_obra = st.text_input(
    "📍 Local da inspeção (ex: Parede Setor 12, Pilar A3...)",
    value=st.session_state.local_input,
    placeholder="Informe o local para salvar no histórico",
)
st.session_state.local_input = local_obra

# Define a imagem ativa (câmera tem prioridade)
img_source = camera_img if camera_img is not None else uploaded
source_name = "camera.jpg" if camera_img else (uploaded.name if uploaded else None)

st.divider()

# ── Área de resultado ──────────────────────────────────────────────────────────
if img_source is not None:
    pil_img   = Image.open(img_source).convert("RGB")
    img_array = np.array(pil_img)

    with st.spinner("🔎 Analisando superfície..."):
        annotated, detections = run_inference(
            model, img_array, conf_thresh, iou_thresh, imgsz
        )

    n_det  = len(detections)
    alerta = n_det > 0
    st.session_state.resultado = {
        "pil_orig": pil_img,
        "annotated": annotated,
        "detections": detections,
        "local": local_obra or source_name,
        "source_name": source_name,
        "n_det": n_det,
    }

res = st.session_state.resultado

if res:
    n_det  = res["n_det"]
    alerta = n_det > 0

    if alerta:
        status_txt = f"INSPEÇÃO CONCLUÍDA — {n_det} RACHADURA(S) DETECTADA(S)"
        detect_txt = f"🚨 DETECÇÃO: RACHADURA IDENTIFICADA — INSPEÇÃO TÉCNICA RECOMENDADA"
        banner_cls = "status-banner-alert"
        detect_cls = "detect-badge-alert"
    else:
        status_txt = "INSPEÇÃO CONCLUÍDA — SUPERFÍCIE ANALISADA"
        detect_txt = "✅ DETECÇÃO: NENHUMA RACHADURA IDENTIFICADA"
        banner_cls = "status-banner-ok"
        detect_cls = "detect-badge-ok"

    st.markdown(f'<div class="{banner_cls}">{status_txt}</div>', unsafe_allow_html=True)

    col_orig, col_res = st.columns(2)
    with col_orig:
        st.markdown('<div class="img-frame">', unsafe_allow_html=True)
        st.image(res["pil_orig"], use_container_width=True, caption="Original")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_res:
        st.markdown('<div class="img-frame">', unsafe_allow_html=True)
        st.image(res["annotated"], use_container_width=True, caption="Detecção IA")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="{detect_cls}">{detect_txt}</div>', unsafe_allow_html=True)

    # Tabela de detecções
    if alerta:
        st.markdown("##### 📊 Detecções")
        st.dataframe(
            [{"#": d["id"], "Classe": d["classe"],
              "Confiança": f"{d['confianca']:.1%}"}
             for d in res["detections"]],
            use_container_width=True, hide_index=True,
        )

    st.markdown("")

    # Observação
    if st.session_state.get("show_obs"):
        obs = st.text_area("✏️ Observação técnica", value=st.session_state.obs_texto,
                           placeholder="Ex: Fissura estrutural, horizontal, ~30cm...")
        st.session_state.obs_texto = obs

    # Botões de ação
    btn1, btn2, btn3 = st.columns(3)

    with btn1:
        if st.button("💾  SALVAR REGISTRO", type="primary", use_container_width=True):
            thumb_b64 = pil_to_b64(res["pil_orig"])
            entry = {
                "local":  res["local"] or "Sem identificação",
                "data":   datetime.now().strftime("%d/%m  %H:%M"),
                "n_det":  res["n_det"],
                "obs":    st.session_state.obs_texto,
                "thumb":  thumb_b64,
            }
            st.session_state.historico.append(entry)
            save_historico(st.session_state.historico)
            st.success("✅ Registro salvo no diário de obra!")
            st.rerun()

    with btn2:
        if st.button("✏️  ADICIONAR OBSERVAÇÃO", use_container_width=True):
            st.session_state.show_obs = not st.session_state.get("show_obs", False)
            st.rerun()

    with btn3:
        if st.button("🔄  NOVA INSPEÇÃO", use_container_width=True):
            st.session_state.resultado  = None
            st.session_state.obs_texto  = ""
            st.session_state.local_input = ""
            st.session_state.show_obs   = False
            st.rerun()

    # Download
    st.markdown("")
    buf = io.BytesIO()
    Image.fromarray(res["annotated"]).save(buf, format="JPEG", quality=95)
    st.download_button(
        "⬇️ Baixar imagem anotada",
        data=buf.getvalue(),
        file_name=f"inspecao_{res['source_name']}",
        mime="image/jpeg",
        use_container_width=True,
    )

else:
    # Estado vazio — guia de uso
    st.markdown("""
    <div style="text-align:center; padding: 40px 20px; color:#888;">
      <div style="font-size:4rem;">📷</div>
      <div style="font-size:1.1rem; font-weight:600; margin:12px 0 8px 0; color:#444;">
        Envie uma foto ou use a câmera acima
      </div>
      <div style="font-size:0.85rem;">
        O modelo detecta automaticamente rachaduras e fissuras na superfície.
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 Dicas para boas fotos em campo"):
        st.markdown("""
| ✅ Recomendado | ❌ Evitar |
|---|---|
| Luz natural ou iluminação uniforme | Flash direto (reflexo apaga detalhes) |
| Câmera perpendicular à parede | Ângulo muito inclinado |
| Distância 50–150 cm | Muito longe (perde detalhe das fissuras) |
| Imagem nítida e focada | Foto tremida ou desfocada |
| Parede seca | Parede molhada (reduz contraste) |
| Alta resolução (câmera do celular) | Capturas comprimidas de baixa qualidade |
""")
