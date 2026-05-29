"""
Inspetor de Qualidade — Detecção de Rachaduras em Paredes
Interface profissional para inspeção em canteiro de obras.

Uso:
    streamlit run app/app.py
"""

import base64
import csv
import io
import json
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ── Constantes ─────────────────────────────────────────────────────────────────
# Configurações fixas — removidas da interface para simplificar uso em campo
MODEL_PATH  = "models/best_v3_nano_1280.pt"
CONF        = 0.15
IOU         = 0.45
IMGSZ       = 1280

CARGOS = ["Engenheiro(a)", "Mestre de Obras", "Técnico(a)", "Inspetor(a)", "Outro"]
HISTORICO_FILE = Path("results/historico_inspecoes.json")
HISTORICO_FILE.parent.mkdir(exist_ok=True)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Registro de Rachaduras e Fissuras",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Esconde barra nativa do Streamlit */
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { display: none !important; }
footer { visibility: hidden; }

.block-container { padding-top: 1rem !important; }
[data-testid="stSidebar"] > div:first-child { padding-top: 0rem; }

.header-bar {
    background: linear-gradient(90deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    color: white;
    padding: 16px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 4px solid #e94560;
    border-radius: 10px;
    margin-bottom: 1.2rem;
}
.header-title {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 2px;
    color: white;
    text-transform: uppercase;
}
.header-icons { font-size: 1.4rem; }

.logo-box {
    background: #1a1a2e;
    color: white;
    text-align: center;
    padding: 16px 8px 10px 8px;
    border-radius: 10px;
    margin-bottom: 10px;
    border: 1px solid #333;
}
.logo-box .logo-icon { font-size: 2.2rem; }
.logo-box .logo-text {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #aab;
    text-transform: uppercase;
    margin-top: 4px;
}

.colab-box {
    background: #f0f4ff;
    border: 1px solid #c5d0f0;
    border-radius: 8px;
    padding: 10px 12px 8px 12px;
    margin-bottom: 10px;
}
.colab-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 4px;
}
.colab-name {
    font-size: 0.9rem;
    font-weight: 700;
    color: #1a1a2e;
}
.colab-cargo {
    font-size: 0.75rem;
    color: #0f3460;
    font-weight: 600;
}

.section-title {
    background: #1a1a2e;
    color: #dde;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 7px 12px;
    border-radius: 6px;
    margin: 10px 0 8px 0;
}

/* Sidebar — textos maiores */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stSelectbox label {
    font-size: 0.88rem !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { font-size: 0.85rem; }

.hist-card {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 8px 10px;
    margin-bottom: 6px;
}
.hist-card.alert { border-left: 4px solid #e94560; }
.hist-card.ok    { border-left: 4px solid #28a745; }
.hist-thumb { width: 52px; height: 40px; object-fit: cover; border-radius: 4px; flex-shrink:0; }
.hist-info  { flex: 1; min-width: 0; }
.hist-name  { font-size: 0.82rem; font-weight: 700; color: #222; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hist-date  { font-size: 0.74rem; color: #666; margin-top: 2px; }
.hist-colab { font-size: 0.72rem; color: #0f3460; font-style: italic; margin-top: 1px; }

/* Reduz altura da área de câmera e upload */
[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img { max-height: 200px !important; }
[data-testid="stCameraInputButton"] { padding: 6px 12px !important; font-size: 0.8rem !important; }
section[data-testid="stFileUploadDropzone"] {
    padding: 12px 16px !important;
    min-height: 80px !important;
}
section[data-testid="stFileUploadDropzone"] > div {
    font-size: 0.82rem !important;
}
[data-testid="stCameraInput"] { max-height: 220px; overflow: hidden; }

.status-banner-ok {
    background: linear-gradient(90deg, #1a1a2e, #0f3460);
    color: white; text-align: center; padding: 8px;
    border-radius: 8px 8px 0 0;
    font-size: 0.8rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;
}
.status-banner-alert {
    background: linear-gradient(90deg, #7b1e1e, #e94560);
    color: white; text-align: center; padding: 8px;
    border-radius: 8px 8px 0 0;
    font-size: 0.8rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;
}
.detect-badge-alert {
    background: linear-gradient(90deg, #7b1e1e, #e94560);
    color: white; text-align: center; padding: 8px;
    border-radius: 0 0 8px 8px;
    font-size: 0.8rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;
}
.detect-badge-ok {
    background: linear-gradient(90deg, #1a4d2e, #28a745);
    color: white; text-align: center; padding: 8px;
    border-radius: 0 0 8px 8px;
    font-size: 0.8rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;
}
.img-frame { border: 2px solid #1a1a2e; border-top: none; border-bottom: none; }

div[data-testid="column"] .stButton > button {
    width: 100%;
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-radius: 6px;
    padding: 10px 4px;
    border: none;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando modelo...")
def load_model(path: str) -> YOLO:
    return YOLO(path)


def pil_to_b64(img: Image.Image, size=(60, 44)) -> str:
    img_s = img.copy(); img_s.thumbnail(size)
    buf = io.BytesIO(); img_s.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def run_inference(model, image, conf, iou, imgsz):
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


def historico_to_csv(hist: list) -> bytes:
    buf = io.StringIO()
    campos = ["data", "local", "colaborador", "cargo", "n_rachaduras", "observacao"]
    writer = csv.DictWriter(buf, fieldnames=campos, extrasaction="ignore")
    writer.writeheader()
    for item in hist:
        writer.writerow({
            "data":          item.get("data", ""),
            "local":         item.get("local", ""),
            "colaborador":   item.get("colaborador", ""),
            "cargo":         item.get("cargo", ""),
            "n_rachaduras":  item.get("n_det", 0),
            "observacao":    item.get("obs", ""),
        })
    return buf.getvalue().encode("utf-8-sig")   # BOM para abrir direto no Excel


# ── Session state ──────────────────────────────────────────────────────────────
for key, val in [("historico", load_historico()), ("resultado", None),
                 ("obs_texto", ""), ("local_input", ""),
                 ("show_obs", False), ("del_confirm", -1)]:
    if key not in st.session_state:
        st.session_state[key] = val


# ── Cabeçalho ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <div class="header-icons">☰</div>
  <div class="header-title">🏗️ &nbsp; Registro de Rachaduras e Fissuras</div>
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

    # ── Colaborador ──────────────────────────────────────────────────────────
    with st.expander("👷 Colaborador", expanded=True):
        nome_colab  = st.text_input("Nome completo", placeholder="Ex: João Silva",
                                    key="nome_colab")
        cargo_colab = st.selectbox("Cargo", CARGOS, key="cargo_colab")

    # Exibe badge do colaborador ativo
    if nome_colab:
        st.markdown(f"""
        <div class="colab-box">
          <div class="colab-label">👤 Responsável pelo registro</div>
          <div class="colab-name">{nome_colab}</div>
          <div class="colab-cargo">{cargo_colab}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Histórico ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Diário de Obra (Histórico)</div>',
                unsafe_allow_html=True)

    hist = st.session_state.historico
    recentes = list(enumerate(hist))[-10:]   # índice global + item

    if not recentes:
        st.caption("Nenhuma inspeção salva ainda.")
    else:
        for idx, item in reversed(recentes):
            alerta    = item["n_det"] > 0
            cls_card  = "alert" if alerta else "ok"
            badge     = "❗" if alerta else "✅"
            thumb     = item.get("thumb", "")
            thumb_html = (f'<img class="hist-thumb" src="data:image/jpeg;base64,{thumb}">'
                          if thumb else "<span style='font-size:1.5rem'>📷</span>")
            status_txt = f"{item['n_det']} rachadura(s)" if alerta else "Sem falhas"
            colab_txt  = item.get("colaborador", "")

            col_card, col_del = st.columns([5, 1])
            with col_card:
                st.markdown(f"""
                <div class="hist-card {cls_card}">
                  {thumb_html}
                  <div class="hist-info">
                    <div class="hist-name">{item['local'][:22]}</div>
                    <div class="hist-date">{item['data']} · {status_txt}</div>
                    {"<div class='hist-colab'>" + colab_txt + "</div>" if colab_txt else ""}
                  </div>
                  <span>{badge}</span>
                </div>
                """, unsafe_allow_html=True)
            with col_del:
                if st.session_state.del_confirm == idx:
                    # Confirmação de exclusão
                    if st.button("✓", key=f"ok_{idx}", help="Confirmar exclusão"):
                        st.session_state.historico.pop(idx)
                        save_historico(st.session_state.historico)
                        st.session_state.del_confirm = -1
                        st.rerun()
                    if st.button("✗", key=f"no_{idx}", help="Cancelar"):
                        st.session_state.del_confirm = -1
                        st.rerun()
                else:
                    if st.button("🗑", key=f"del_{idx}", help="Excluir este registro"):
                        st.session_state.del_confirm = idx
                        st.rerun()

    st.markdown("")
    # Botão limpar tudo
    if hist:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Limpar tudo", use_container_width=True):
                st.session_state.historico = []
                save_historico([])
                st.rerun()
        with c2:
            csv_bytes = historico_to_csv(hist)
            st.download_button(
                "📥 CSV",
                data=csv_bytes,
                file_name=f"diario_obra_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Exportar histórico como planilha Excel/CSV",
            )


# ── Conteúdo principal ─────────────────────────────────────────────────────────
if not Path(MODEL_PATH).exists():
    st.error(f"Modelo não encontrado: `{MODEL_PATH}`. Execute o treino primeiro.")
    st.stop()

model = load_model(MODEL_PATH)

col_up, col_cam = st.columns(2)
with col_up:
    st.markdown("#### 📁 Enviar foto")
    uploaded = st.file_uploader("foto", type=["jpg","jpeg","png","bmp","webp"],
                                 label_visibility="collapsed")
with col_cam:
    st.markdown("#### 📷 Tirar foto")
    camera_img = st.camera_input("cam", label_visibility="collapsed")

local_obra = st.text_input(
    "📍 Local da inspeção",
    value=st.session_state.local_input,
    placeholder="Ex: Parede Setor 12, Pilar A3, Laje Bloco B...",
)
st.session_state.local_input = local_obra

img_source  = camera_img if camera_img is not None else uploaded
source_name = "camera.jpg" if camera_img else (uploaded.name if uploaded else None)

st.divider()

# ── Inferência e resultado ─────────────────────────────────────────────────────
if img_source is not None:
    pil_img   = Image.open(img_source).convert("RGB")
    img_array = np.array(pil_img)
    with st.spinner("🔎 Analisando superfície..."):
        annotated, detections = run_inference(model, img_array, CONF, IOU, IMGSZ)
    st.session_state.resultado = {
        "pil_orig": pil_img, "annotated": annotated,
        "detections": detections,
        "local": local_obra or source_name,
        "source_name": source_name,
        "n_det": len(detections),
    }

res = st.session_state.resultado

if res:
    n_det  = res["n_det"]
    alerta = n_det > 0

    status_txt  = (f"INSPEÇÃO CONCLUÍDA — {n_det} RACHADURA(S) DETECTADA(S)"
                   if alerta else "INSPEÇÃO CONCLUÍDA — SUPERFÍCIE ANALISADA")
    detect_txt  = ("🚨 DETECÇÃO: RACHADURA IDENTIFICADA — INSPEÇÃO TÉCNICA RECOMENDADA"
                   if alerta else "✅ DETECÇÃO: NENHUMA RACHADURA IDENTIFICADA")
    banner_cls  = "status-banner-alert" if alerta else "status-banner-ok"
    detect_cls  = "detect-badge-alert"  if alerta else "detect-badge-ok"

    st.markdown(f'<div class="{banner_cls}">{status_txt}</div>', unsafe_allow_html=True)

    col_orig, col_res = st.columns(2)
    with col_orig:
        st.image(res["pil_orig"],  use_container_width=True, caption="Original")
    with col_res:
        st.image(res["annotated"], use_container_width=True, caption="Detecção IA")

    st.markdown(f'<div class="{detect_cls}">{detect_txt}</div>', unsafe_allow_html=True)

    if alerta:
        st.markdown("##### 📊 Detecções")
        st.dataframe(
            [{"#": d["id"], "Classe": d["classe"], "Confiança": f"{d['confianca']:.1%}"}
             for d in res["detections"]],
            use_container_width=True, hide_index=True,
        )

    st.markdown("")

    if st.session_state.show_obs:
        obs = st.text_area("✏️ Observação técnica", value=st.session_state.obs_texto,
                           placeholder="Ex: Fissura estrutural, horizontal, ~30cm...")
        st.session_state.obs_texto = obs

    # ── Botões de ação ─────────────────────────────────────────────────────
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("💾  SALVAR REGISTRO", type="primary", use_container_width=True):
            if not nome_colab:
                st.warning("⚠️ Informe o nome do colaborador antes de salvar.")
            else:
                entry = {
                    "local":       res["local"] or "Sem identificação",
                    "data":        datetime.now().strftime("%d/%m/%Y %H:%M"),
                    "colaborador": nome_colab,
                    "cargo":       cargo_colab,
                    "n_det":       res["n_det"],
                    "obs":         st.session_state.obs_texto,
                    "thumb":       pil_to_b64(res["pil_orig"]),
                }
                st.session_state.historico.append(entry)
                save_historico(st.session_state.historico)
                st.success(f"✅ Registro salvo por {nome_colab} ({cargo_colab})!")
                st.rerun()
    with b2:
        if st.button("✏️  ADICIONAR OBSERVAÇÃO", use_container_width=True):
            st.session_state.show_obs = not st.session_state.show_obs
            st.rerun()
    with b3:
        if st.button("🔄  NOVA INSPEÇÃO", use_container_width=True):
            st.session_state.resultado   = None
            st.session_state.obs_texto   = ""
            st.session_state.local_input = ""
            st.session_state.show_obs    = False
            st.rerun()

    st.divider()

    # ── Downloads ──────────────────────────────────────────────────────────
    d1, d2 = st.columns(2)
    with d1:
        buf = io.BytesIO()
        Image.fromarray(res["annotated"]).save(buf, format="JPEG", quality=95)
        st.download_button(
            "⬇️ Baixar imagem anotada",
            data=buf.getvalue(),
            file_name=f"inspecao_{res['source_name']}",
            mime="image/jpeg",
            use_container_width=True,
        )
    with d2:
        if st.session_state.historico:
            st.download_button(
                "📊 Exportar histórico CSV",
                data=historico_to_csv(st.session_state.historico),
                file_name=f"diario_obra_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Abre direto no Excel",
            )

else:
    st.markdown("""
    <div style="text-align:center;padding:40px 20px;color:#888;">
      <div style="font-size:4rem;">📷</div>
      <div style="font-size:1.1rem;font-weight:600;margin:12px 0 8px 0;color:#444;">
        Envie uma foto ou use a câmera acima
      </div>
      <div style="font-size:0.85rem;">O modelo detecta rachaduras e fissuras automaticamente.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 Dicas para boas fotos em campo"):
        st.markdown("""
| ✅ Recomendado | ❌ Evitar |
|---|---|
| Luz natural ou iluminação uniforme | Flash direto (reflexo apaga detalhes) |
| Câmera perpendicular à parede | Ângulo muito inclinado |
| Distância 50–150 cm da superfície | Muito longe (perde detalhe das fissuras) |
| Imagem nítida e focada | Foto tremida ou desfocada |
| Parede seca | Parede molhada (reduz contraste) |
| Alta resolução (câmera do celular) | Capturas de baixa qualidade |
""")
