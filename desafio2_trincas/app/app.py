"""
Inspetor de Qualidade - Deteccao de rachaduras e fissuras.

Uso:
    streamlit run app/app.py
"""

import base64
import csv
import html
import io
import json
import re
import unicodedata
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = "models/best_v3_nano_1280.pt"
CONF = 0.15
IOU = 0.45
IMGSZ = 640
DEVICE = 0 if torch.cuda.is_available() else "cpu"

CARGOS = ["Engenheiro(a)", "Mestre de Obras", "Tecnico(a)", "Inspetor(a)", "Outro"]
HISTORICO_FILE = Path("results/historico_inspecoes.json")
HISTORICO_FILE.parent.mkdir(exist_ok=True)


st.set_page_config(
    page_title="Registro de Rachaduras e Fissuras",
    page_icon="RI",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
:root {
    --bg: #050b14;
    --panel: #0b1422;
    --panel-2: #101a2b;
    --stroke: rgba(148, 163, 184, .22);
    --stroke-strong: rgba(255, 77, 93, .58);
    --text: #f8fafc;
    --muted: #a8b3c7;
    --red: #ef4454;
    --red-2: #b91f34;
    --green: #22c55e;
    --blue: #6aa6ff;
}

html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(239, 68, 84, .10), transparent 28rem),
        linear-gradient(180deg, #07101d 0%, #030812 100%) !important;
    color: var(--text);
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    display: none !important;
}
footer { visibility: hidden; }
.block-container {
    max-width: 520px;
    padding: .7rem .75rem 6.2rem !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07101d 0%, #050914 100%);
    border-right: 1px solid var(--stroke);
}

h1, h2, h3, h4, p, label, span, div { color: inherit; }
h1, h2, h3 { letter-spacing: -.02em; }

.app-shell {
    max-width: 430px;
    margin: 0 auto;
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 34px;
    background:
        linear-gradient(180deg, rgba(9, 19, 34, .96), rgba(3, 8, 18, .98)),
        radial-gradient(circle at 60% 0%, rgba(41, 92, 169, .20), transparent 22rem);
    box-shadow: 0 26px 80px rgba(0,0,0,.42);
    padding: 1.05rem;
}

.app-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: .9rem;
    margin-bottom: 1.15rem;
}
.brand {
    display: flex;
    align-items: center;
    gap: .72rem;
}
.brand-mark {
    width: 42px;
    height: 42px;
    border-radius: 14px;
    display: grid;
    place-items: center;
    color: white;
    background: linear-gradient(135deg, var(--red), #ff7a45);
    box-shadow: 0 10px 26px rgba(239, 68, 84, .24);
}
.brand-mark svg,
.cloud-pill svg,
.section-label .icon svg,
.status-icon svg,
.alert-mark svg {
    width: 1.12rem;
    height: 1.12rem;
    stroke: currentColor;
    stroke-width: 2.2;
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
}
.brand-title {
    font-weight: 850;
    font-size: 1.02rem;
    line-height: 1.18;
}
.brand-sub {
    color: var(--muted);
    font-size: .75rem;
    margin-top: .16rem;
}
.cloud-pill {
    width: 40px;
    height: 40px;
    border-radius: 999px;
    display: grid;
    place-items: center;
    border: 1px solid var(--stroke-strong);
    color: var(--red);
    background: rgba(239, 68, 84, .08);
}

.section-label {
    display: flex;
    align-items: center;
    gap: .55rem;
    color: var(--text);
    font-weight: 800;
    font-size: 1.05rem;
    margin: 1.1rem 0 .68rem;
}
.section-label .icon {
    color: var(--red);
    border: 1px solid rgba(239,68,84,.46);
    border-radius: 999px;
    width: 30px;
    height: 30px;
    display: inline-grid;
    place-items: center;
    background: rgba(239,68,84,.08);
}

.soft-card {
    border: 1px solid var(--stroke);
    border-radius: 13px;
    background: linear-gradient(180deg, rgba(20, 31, 49, .96), rgba(13, 23, 39, .92));
    padding: .85rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.04);
}
.identity-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: .85rem;
    border: 1px solid var(--stroke);
    border-radius: 13px;
    background: linear-gradient(180deg, rgba(28, 40, 62, .95), rgba(13, 22, 37, .95));
    padding: .78rem .85rem;
    margin: .45rem 0 .7rem;
}
.avatar {
    width: 45px;
    height: 45px;
    border-radius: 999px;
    display: grid;
    place-items: center;
    background: radial-gradient(circle at 30% 30%, #fff, #ffd6d9 42%, var(--red) 43%);
    color: var(--red-2);
    font-weight: 900;
}
.selected-pill {
    color: #8df0b1;
    background: rgba(34,197,94,.14);
    border: 1px solid rgba(34,197,94,.25);
    border-radius: 999px;
    padding: .36rem .62rem;
    font-weight: 800;
    font-size: .75rem;
    white-space: nowrap;
}

.status-card {
    display: flex;
    align-items: center;
    gap: .9rem;
    border-radius: 14px;
    padding: 1rem;
    margin: .7rem 0 1rem;
    background: linear-gradient(135deg, #9f1d31 0%, var(--red) 100%);
    box-shadow: 0 14px 34px rgba(239,68,84,.25);
}
.status-card.ok {
    background: linear-gradient(135deg, #0f5132 0%, #22c55e 100%);
}
.status-icon {
    width: 46px;
    height: 46px;
    border-radius: 999px;
    border: 2px solid rgba(255,255,255,.38);
    display: grid;
    place-items: center;
    font-size: 1.25rem;
}
.status-title { font-size: 1.05rem; font-weight: 900; }
.status-sub { color: rgba(255,255,255,.82); font-size: .88rem; margin-top: .12rem; }

.metric-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: .7rem;
    margin: .85rem 0;
}
.metric-card {
    border: 1px solid var(--stroke);
    border-radius: 13px;
    padding: .82rem;
    background: rgba(16, 26, 43, .88);
}
.metric-value {
    color: var(--text);
    font-size: 1.65rem;
    font-weight: 900;
    line-height: 1;
}
.metric-label {
    color: var(--muted);
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-top: .45rem;
}

.hist-card {
    display: flex;
    align-items: center;
    gap: .8rem;
    border: 1px solid var(--stroke);
    border-left: 4px solid var(--red);
    border-radius: 12px;
    padding: .72rem;
    margin-bottom: .62rem;
    background: linear-gradient(180deg, rgba(20,31,49,.96), rgba(12,21,36,.95));
}
.hist-card.ok { border-left-color: var(--green); }
.hist-thumb {
    width: 64px;
    height: 54px;
    object-fit: cover;
    border-radius: 9px;
    border: 1px solid rgba(255,255,255,.12);
    flex-shrink: 0;
}
.hist-title {
    font-weight: 850;
    font-size: .94rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.hist-meta { color: var(--muted); font-size: .8rem; margin-top: .16rem; }
.hist-person { color: #93c5fd; font-size: .82rem; font-style: italic; margin-top: .18rem; }
.alert-mark { color: var(--red); font-size: 1.35rem; margin-left: auto; font-weight: 900; }

.image-frame img {
    border-radius: 13px !important;
    border: 1px solid var(--stroke);
}
.help-empty {
    text-align: center;
    border: 1px dashed rgba(239,68,84,.55);
    border-radius: 16px;
    padding: 2rem 1rem;
    background: rgba(239,68,84,.055);
    color: var(--muted);
}

.bottom-nav {
    position: fixed;
    left: 50%;
    bottom: 16px;
    transform: translateX(-50%);
    width: min(460px, calc(100vw - 26px));
    z-index: 999;
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 24px;
    background: rgba(4, 10, 20, .92);
    box-shadow: 0 12px 40px rgba(0,0,0,.38);
    backdrop-filter: blur(14px);
    padding: .38rem .55rem .2rem;
}

div[role="radiogroup"] {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: .2rem;
}
div[role="radiogroup"] label {
    border-radius: 16px;
    padding: .42rem .1rem .5rem !important;
    justify-content: center;
    text-align: center;
}
div[role="radiogroup"] label:has(input:checked) {
    background: rgba(239,68,84,.16);
    color: white !important;
}

.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stTextArea textarea,
.stNumberInput input {
    background: #162237 !important;
    border: 1px solid rgba(148,163,184,.25) !important;
    border-radius: 11px !important;
    color: var(--text) !important;
}
.stFileUploader section {
    min-height: 150px;
    border: 1.5px dashed rgba(239,68,84,.65) !important;
    border-radius: 16px !important;
    background: rgba(239,68,84,.05) !important;
}
[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img {
    border-radius: 15px !important;
    border: 1px solid var(--stroke);
    max-height: 270px !important;
}
[data-testid="stCameraInputButton"] > button {
    background: linear-gradient(135deg, var(--red), #ff6b76) !important;
    color: white !important;
    border-radius: 999px !important;
    width: 100% !important;
    font-weight: 900 !important;
}
.stButton button,
.stDownloadButton button {
    border-radius: 12px !important;
    border: 1px solid var(--stroke) !important;
    background: rgba(22, 34, 55, .92) !important;
    color: var(--text) !important;
    font-weight: 800 !important;
}
.stButton button[kind="primary"],
.stDownloadButton button[kind="primary"] {
    background: linear-gradient(135deg, var(--red), #ff6673) !important;
    border-color: transparent !important;
}

@media (max-width: 640px) {
    .block-container { padding-left: .7rem !important; padding-right: .7rem !important; }
    .app-shell { border-radius: 24px; padding: .85rem; }
    .metric-row { grid-template-columns: 1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Carregando modelo...")
def load_model(path: str) -> YOLO:
    return YOLO(path)


def esc(value) -> str:
    return html.escape(str(value or ""))


def normalize_text(value) -> str:
    text = str(value or "")
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def parse_data(value: str) -> datetime | None:
    for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value or "", fmt)
        except ValueError:
            continue
    return None


def dia_label(item: dict) -> str:
    dt = parse_data(item.get("data", ""))
    return dt.strftime("%d/%m/%Y") if dt else "Sem data"


def pil_to_b64(img: Image.Image, size=(80, 60)) -> str:
    img_s = img.copy()
    img_s.thumbnail(size)
    buf = io.BytesIO()
    img_s.save(buf, format="JPEG", quality=72)
    return base64.b64encode(buf.getvalue()).decode()


def image_to_download(img_array) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def run_inference(model, image, conf, iou, imgsz):
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=DEVICE,
        verbose=False,
        retina_masks=True,
    )[0]
    annotated_rgb = cv2.cvtColor(results.plot(line_width=3), cv2.COLOR_BGR2RGB)
    detections = []
    if results.boxes is not None:
        for i, box in enumerate(results.boxes):
            detections.append(
                {
                    "id": i + 1,
                    "confianca": float(box.conf[0]),
                    "classe": model.names[int(box.cls[0])],
                }
            )
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
    campos = [
        "data",
        "dia",
        "local",
        "colaborador",
        "cargo",
        "n_rachaduras_ia",
        "contagem_correta",
        "diferenca",
        "observacao",
    ]
    writer = csv.DictWriter(buf, fieldnames=campos, extrasaction="ignore")
    writer.writeheader()
    for item in hist:
        n_ia = int(item.get("n_det", 0) or 0)
        n_cor = int(item.get("contagem_correta", n_ia) or 0)
        writer.writerow(
            {
                "data": item.get("data", ""),
                "dia": dia_label(item),
                "local": item.get("local", ""),
                "colaborador": item.get("colaborador", ""),
                "cargo": item.get("cargo", ""),
                "n_rachaduras_ia": n_ia,
                "contagem_correta": n_cor,
                "diferenca": n_cor - n_ia,
                "observacao": item.get("obs", ""),
            }
        )
    return buf.getvalue().encode("utf-8-sig")


def agrupar_por_dia(hist: list) -> OrderedDict:
    grupos = OrderedDict()
    ordenado = sorted(
        hist,
        key=lambda item: parse_data(item.get("data", "")) or datetime.min,
        reverse=True,
    )
    for item in ordenado:
        grupos.setdefault(dia_label(item), []).append(item)
    return grupos


def resumo(hist: list) -> dict:
    total = len(hist)
    rachaduras = sum(int(i.get("contagem_correta", i.get("n_det", 0)) or 0) for i in hist)
    alertas = sum(1 for i in hist if int(i.get("n_det", 0) or 0) > 0)
    dias = len(agrupar_por_dia(hist))
    return {"total": total, "rachaduras": rachaduras, "alertas": alertas, "dias": dias}


def pdf_escape(text: str) -> str:
    return normalize_text(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def simple_pdf(lines: list[str], title: str = "Relatorio de Inspecoes") -> bytes:
    width, height = 595, 842
    margin = 48
    line_h = 15
    pages = []
    current = []
    for line in lines:
        if len(current) >= 47:
            pages.append(current)
            current = []
        current.append(line)
    pages.append(current or ["Sem registros."])

    objects = []
    font_obj_id = 3 + len(pages) * 2
    kids = []
    next_id = 3

    for page_lines in pages:
        page_id = next_id
        content_id = next_id + 1
        next_id += 2
        kids.append(page_id)
        ops = ["BT", "/F1 16 Tf", f"{margin} {height - margin} Td", f"({pdf_escape(title)}) Tj"]
        ops += ["/F1 10 Tf", f"0 -{line_h * 1.8} Td"]
        for line in page_lines:
            clean = pdf_escape(line[:100])
            ops.append(f"({clean}) Tj")
            ops.append(f"0 -{line_h} Td")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1", errors="replace")
        objects.append((page_id, f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources << /Font << /F1 {font_obj_id} 0 R >> >> /Contents {content_id} 0 R >>".encode()))
        objects.append((content_id, b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"))

    pages_obj = f"<< /Type /Pages /Kids [{' '.join(f'{kid} 0 R' for kid in kids)}] /Count {len(kids)} >>".encode()
    font_obj = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    all_objects = [(1, b"<< /Type /Catalog /Pages 2 0 R >>"), (2, pages_obj), *objects, (font_obj_id, font_obj)]

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = [0]
    for obj_id, body in sorted(all_objects, key=lambda item: item[0]):
        offsets.append(out.tell())
        out.write(f"{obj_id} 0 obj\n".encode())
        out.write(body)
        out.write(b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(offsets)}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.write(f"{offset:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode())
    return out.getvalue()


def relatorio_pdf(hist: list, titulo: str) -> bytes:
    s = resumo(hist)
    lines = [
        f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        f"Total de registros: {s['total']}",
        f"Inspecoes com alerta: {s['alertas']}",
        f"Rachaduras registradas: {s['rachaduras']}",
        f"Dias acompanhados: {s['dias']}",
        "",
    ]
    for dia, itens in agrupar_por_dia(hist).items():
        total_dia = sum(int(i.get("contagem_correta", i.get("n_det", 0)) or 0) for i in itens)
        lines.append(f"{dia} - {len(itens)} registro(s), {total_dia} rachadura(s)")
        for item in itens:
            lines.append(
                f"  {item.get('data','')} | {item.get('local','')} | "
                f"{item.get('colaborador','')} | {item.get('cargo','')} | "
                f"{item.get('contagem_correta', item.get('n_det', 0))} rachadura(s)"
            )
        lines.append("")
    return simple_pdf(lines, titulo)


ICONS = {
    "crane": '<svg viewBox="0 0 24 24"><path d="M4 20h16"/><path d="M6 20V8h8"/><path d="M6 8l8-4 5 4"/><path d="M14 4v16"/><path d="M19 8v4"/><path d="M17 12h4"/></svg>',
    "upload": '<svg viewBox="0 0 24 24"><path d="M12 16V4"/><path d="M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/></svg>',
    "user": '<svg viewBox="0 0 24 24"><path d="M20 21a8 8 0 0 0-16 0"/><circle cx="12" cy="7" r="4"/></svg>',
    "folder": '<svg viewBox="0 0 24 24"><path d="M3 7a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M12 17V10"/><path d="M9 13l3-3 3 3"/></svg>',
    "camera": '<svg viewBox="0 0 24 24"><path d="M4 8h4l2-3h4l2 3h4v11H4z"/><circle cx="12" cy="13" r="4"/></svg>',
    "pin": '<svg viewBox="0 0 24 24"><path d="M12 21s7-5.3 7-12a7 7 0 1 0-14 0c0 6.7 7 12 7 12z"/><circle cx="12" cy="9" r="2.5"/></svg>',
    "image": '<svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="8" cy="10" r="1.6"/><path d="M21 16l-5-5-4 4-2-2-5 5"/></svg>',
    "ia": '<svg viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" rx="3"/><path d="M8 14V8"/><path d="M12 14l2-6 2 6"/><path d="M13 12h2"/></svg>',
    "check": '<svg viewBox="0 0 24 24"><path d="M20 6 9 17l-5-5"/></svg>',
    "clipboard": '<svg viewBox="0 0 24 24"><path d="M9 4h6l1 2h3v15H5V6h3z"/><path d="M9 4v4h6V4"/><path d="M8 12h8"/><path d="M8 16h6"/></svg>',
    "calendar": '<svg viewBox="0 0 24 24"><rect x="4" y="5" width="16" height="15" rx="2"/><path d="M8 3v4"/><path d="M16 3v4"/><path d="M4 10h16"/></svg>',
    "alert": '<svg viewBox="0 0 24 24"><path d="M12 8v5"/><path d="M12 17h.01"/><path d="M10.3 4.5 2.7 18a2 2 0 0 0 1.7 3h15.2a2 2 0 0 0 1.7-3L13.7 4.5a2 2 0 0 0-3.4 0z"/></svg>',
}


def icon_svg(name: str) -> str:
    return ICONS.get(name, ICONS["alert"])


def render_header(subtitle: str):
    st.markdown(
        f"""
        <div class="app-top">
          <div class="brand">
            <div class="brand-mark">{icon_svg("crane")}</div>
            <div>
              <div class="brand-title">Registro de Rachaduras<br>e Fissuras</div>
              <div class="brand-sub">{esc(subtitle)}</div>
            </div>
          </div>
          <div class="cloud-pill">{icon_svg("upload")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, icon: str):
    st.markdown(
        f'<div class="section-label"><span class="icon">{icon_svg(icon)}</span>{esc(title)}</div>',
        unsafe_allow_html=True,
    )


def metric_row(values: list[tuple[str, str]]):
    cards = "".join(
        f'<div class="metric-card"><div class="metric-value">{esc(value)}</div><div class="metric-label">{esc(label)}</div></div>'
        for value, label in values
    )
    st.markdown(f'<div class="metric-row">{cards}</div>', unsafe_allow_html=True)


def hist_card(item: dict):
    alerta = int(item.get("n_det", 0) or 0) > 0
    cls = "" if alerta else " ok"
    thumb = item.get("thumb", "")
    thumb_html = (
        f'<img class="hist-thumb" src="data:image/jpeg;base64,{thumb}">'
        if thumb
        else f'<div class="hist-thumb" style="display:grid;place-items:center;color:var(--red);">{icon_svg("image")}</div>'
    )
    count = item.get("contagem_correta", item.get("n_det", 0))
    mark = icon_svg("alert") if alerta else icon_svg("check")
    st.markdown(
        f"""
        <div class="hist-card{cls}">
          {thumb_html}
          <div style="min-width:0; flex:1;">
            <div class="hist-title">{esc(item.get("local", "Sem local"))}</div>
            <div class="hist-meta">{esc(item.get("data", ""))} &middot; {esc(count)} rachadura(s)</div>
            <div class="hist-person">{esc(item.get("colaborador", ""))}</div>
          </div>
          <div class="alert-mark">{mark}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_history_list(hist: list, manage: bool = False):
    if not hist:
        st.markdown('<div class="help-empty">Nenhuma inspecao salva ainda.</div>', unsafe_allow_html=True)
        return
    for idx, item in reversed(list(enumerate(hist))):
        col_card, col_action = st.columns([8, 1])
        with col_card:
            hist_card(item)
        if manage:
            with col_action:
                if st.session_state.del_confirm == idx:
                    if st.button("OK", key=f"ok_{idx}", help="Confirmar"):
                        st.session_state.historico.pop(idx)
                        save_historico(st.session_state.historico)
                        st.session_state.del_confirm = -1
                        st.rerun()
                    if st.button("X", key=f"no_{idx}", help="Cancelar"):
                        st.session_state.del_confirm = -1
                        st.rerun()
                elif st.button("Del", key=f"del_{idx}", help="Excluir"):
                    st.session_state.del_confirm = idx
                    st.rerun()


def page_inspecao():
    render_header("Inspetor de Qualidade - Engenharia Civil")
    section("Colaborador", "user")

    nome_colab = st.text_input(
        "Nome do colaborador",
        placeholder="Ex.: Raquel",
        key="nome_colab",
    )
    cargo_colab = st.selectbox("Funcao / Cargo", CARGOS, key="cargo_colab")

    if nome_colab:
        initials = "".join(part[:1] for part in nome_colab.split()[:2]).upper() or "R"
        st.markdown(
            f"""
            <div class="identity-card">
              <div style="display:flex;align-items:center;gap:.75rem;min-width:0;">
                <div class="avatar">{esc(initials)}</div>
                <div style="min-width:0;">
                  <div style="font-weight:900;">{esc(nome_colab)}</div>
                  <div style="color:var(--muted);font-size:.84rem;">{esc(cargo_colab)}</div>
                </div>
              </div>
              <div class="selected-pill">Selecionado</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section("Envio de imagem", "folder")
    uploaded = st.file_uploader(
        "Upload de arquivos",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="JPG, PNG, BMP ou WEBP ate 200MB.",
    )

    section("Captura em campo", "camera")
    camera_img = st.camera_input("Abrir camera", label_visibility="collapsed")

    section("Informacoes da inspecao", "pin")
    local_obra = st.text_input(
        "Local da inspecao",
        value=st.session_state.local_input,
        placeholder="Ex.: Ponto de apoio, Pilar 12, Viga 03",
    )
    st.session_state.local_input = local_obra

    img_source = camera_img if camera_img is not None else uploaded
    source_name = "camera.jpg" if camera_img else (uploaded.name if uploaded else None)

    if img_source is not None:
        if not nome_colab:
            st.warning("Informe o colaborador antes de concluir o registro.")

        if not Path(MODEL_PATH).exists():
            st.error(f"Modelo nao encontrado: `{MODEL_PATH}`. Execute o treino primeiro.")
            st.stop()

        pil_img = Image.open(img_source).convert("RGB")
        img_array = np.array(pil_img)
        with st.spinner("Analisando superficie..."):
            model = load_model(MODEL_PATH)
            annotated, detections = run_inference(model, img_array, CONF, IOU, IMGSZ)
        st.session_state.resultado = {
            "pil_orig": pil_img,
            "annotated": annotated,
            "detections": detections,
            "local": local_obra or source_name,
            "source_name": source_name,
            "n_det": len(detections),
            "nome_colab": nome_colab,
            "cargo_colab": cargo_colab,
        }
        st.session_state.page = "Resultado"
        st.rerun()

    st.markdown(
        '<div class="help-empty">Envie uma imagem ou abra a camera para iniciar a analise por IA.</div>',
        unsafe_allow_html=True,
    )


def page_resultado():
    render_header("Resultado da inspecao")
    res = st.session_state.resultado
    if not res:
        st.markdown('<div class="help-empty">Nenhuma inspecao analisada ainda.</div>', unsafe_allow_html=True)
        if st.button("Iniciar coleta", type="primary", width="stretch"):
            st.session_state.page = "Inicio"
            st.rerun()
        return

    n_det = int(res["n_det"])
    alerta = n_det > 0
    st.markdown(
        f"""
        <div class="status-card {'ok' if not alerta else ''}">
          <div class="status-icon">{icon_svg("check")}</div>
          <div>
            <div class="status-title">Inspecao concluida</div>
            <div class="status-sub">{n_det} rachadura(s) detectada(s)</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section("Imagem original", "image")
    st.markdown('<div class="image-frame">', unsafe_allow_html=True)
    st.image(res["pil_orig"], width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    section("Deteccao por IA", "ia")
    st.markdown('<div class="image-frame">', unsafe_allow_html=True)
    st.image(res["annotated"], width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    confidence = 0
    if res["detections"]:
        confidence = max(d["confianca"] for d in res["detections"])
    metric_row(
        [
            (f"{round(confidence * 100)}%", "Confianca"),
            (str(n_det), "IA"),
            (res.get("local") or "Sem local", "Local"),
        ]
    )

    contagem_correta = st.number_input(
        "Contagem correta de rachaduras",
        min_value=0,
        max_value=50,
        value=n_det,
        step=1,
        help="Ajuste se a contagem humana diferir da IA.",
        key="contagem_correta",
    )

    if st.session_state.show_obs:
        st.session_state.obs_texto = st.text_area(
            "Observacao tecnica",
            value=st.session_state.obs_texto,
            placeholder="Ex.: fissura vertical fina proxima ao pilar.",
        )

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Salvar resultado", type="primary", width="stretch"):
            nome_colab = res.get("nome_colab") or st.session_state.get("nome_colab", "")
            if not nome_colab:
                st.warning("Informe o nome do colaborador antes de salvar.")
            else:
                entry = {
                    "local": res["local"] or "Sem identificacao",
                    "data": datetime.now().strftime("%d/%m/%Y %H:%M"),
                    "colaborador": nome_colab,
                    "cargo": res.get("cargo_colab") or st.session_state.get("cargo_colab", ""),
                    "n_det": n_det,
                    "contagem_correta": contagem_correta,
                    "obs": st.session_state.obs_texto,
                    "thumb": pil_to_b64(res["pil_orig"]),
                }
                st.session_state.historico.append(entry)
                save_historico(st.session_state.historico)
                st.success("Registro salvo no diario de obra.")
    with b2:
        if st.button("Adicionar observacao", width="stretch"):
            st.session_state.show_obs = not st.session_state.show_obs
            st.rerun()

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Baixar imagem IA",
            data=image_to_download(res["annotated"]),
            file_name=f"inspecao_{res['source_name'] or 'resultado'}.jpg",
            mime="image/jpeg",
            width="stretch",
        )
    with d2:
        if st.button("Nova inspecao", width="stretch"):
            st.session_state.resultado = None
            st.session_state.obs_texto = ""
            st.session_state.local_input = ""
            st.session_state.show_obs = False
            st.session_state.page = "Inicio"
            st.rerun()


def page_historico():
    render_header("Historico e diario de obra")
    hist = st.session_state.historico
    s = resumo(hist)
    metric_row(
        [
            (str(s["total"]), "Registros"),
            (str(s["rachaduras"]), "Rachaduras"),
            (str(s["dias"]), "Dias"),
        ]
    )
    section("Diario de obra", "clipboard")
    show_history_list(hist, manage=True)

    if hist:
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Exportar CSV",
                data=historico_to_csv(hist),
                file_name=f"diario_obra_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                width="stretch",
            )
        with c2:
            if st.button("Limpar tudo", width="stretch"):
                st.session_state.historico = []
                save_historico([])
                st.rerun()


def page_dashboard():
    render_header("Dashboard de registros diarios")
    hist = st.session_state.historico
    grupos = agrupar_por_dia(hist)
    s = resumo(hist)
    metric_row(
        [
            (str(s["total"]), "Inspecoes"),
            (str(s["alertas"]), "Alertas"),
            (str(s["rachaduras"]), "Rachaduras"),
        ]
    )

    if not hist:
        st.markdown('<div class="help-empty">Sem dados para o dashboard.</div>', unsafe_allow_html=True)
        return

    dias = list(grupos.keys())
    selected_day = st.selectbox("Separar por dia", ["Todos os dias", *dias])
    hist_filtrado = hist if selected_day == "Todos os dias" else grupos[selected_day]

    chart_rows = []
    for dia, itens in reversed(list(grupos.items())):
        chart_rows.append(
            {
                "dia": dia,
                "inspecoes": len(itens),
                "rachaduras": sum(int(i.get("contagem_correta", i.get("n_det", 0)) or 0) for i in itens),
            }
        )
    st.bar_chart(pd.DataFrame(chart_rows), x="dia", y=["inspecoes", "rachaduras"], width="stretch")

    section("Registros do periodo", "calendar")
    show_history_list(hist_filtrado, manage=False)

    safe_day = re.sub(r"[^0-9A-Za-z_-]+", "_", normalize_text(selected_day).strip("_")) or "todos"
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Baixar relatorio PDF",
            data=relatorio_pdf(hist_filtrado, f"Relatorio - {selected_day}"),
            file_name=f"relatorio_inspecoes_{safe_day}.pdf",
            mime="application/pdf",
            type="primary",
            width="stretch",
        )
    with c2:
        st.download_button(
            "Baixar CSV",
            data=historico_to_csv(hist_filtrado),
            file_name=f"relatorio_inspecoes_{safe_day}.csv",
            mime="text/csv",
            width="stretch",
        )


for key, val in [
    ("historico", load_historico()),
    ("resultado", None),
    ("obs_texto", ""),
    ("local_input", ""),
    ("show_obs", False),
    ("del_confirm", -1),
    ("page", "Inicio"),
]:
    if key not in st.session_state:
        st.session_state[key] = val


with st.sidebar:
    st.markdown("### Residencia IA")
    st.caption("Inspetor de Qualidade - Engenharia Civil")
    st.divider()
    st.markdown("#### Ultimas inspecoes")
    show_history_list(st.session_state.historico[-5:], manage=False)


st.markdown('<div class="app-shell">', unsafe_allow_html=True)

page = st.session_state.page
if page == "Inicio":
    page_inspecao()
elif page == "Resultado":
    page_resultado()
elif page == "Historico":
    page_historico()
else:
    page_dashboard()

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="bottom-nav">', unsafe_allow_html=True)
selected_page = st.radio(
    "Navegacao",
    ["Inicio", "Historico", "Resultado", "Dashboard"],
    index=["Inicio", "Historico", "Resultado", "Dashboard"].index(st.session_state.page),
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("</div>", unsafe_allow_html=True)

if selected_page != st.session_state.page:
    st.session_state.page = selected_page
    st.rerun()

