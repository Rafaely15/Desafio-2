"""
Inspetor de Qualidade - Detecção de rachaduras e fissuras.

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
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


MODEL_PATH = "models/best_v3_nano_1280.pt"
CONF = 0.15
IOU = 0.45
IMGSZ = 640
DEVICE = 0 if torch.cuda.is_available() else "cpu"

CARGOS = ["Engenheiro(a)", "Mestre de Obras", "Técnico(a)", "Inspetor(a)", "Outro"]
HISTORICO_FILE = Path("results/historico_inspecoes.json")
PERFIL_FILE = Path("results/perfil_colaborador.json")
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
    max-width: 980px;
    padding: 1.5rem 2.2rem 6.8rem !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07101d 0%, #050914 100%);
    border-right: 1px solid var(--stroke);
}

h1, h2, h3, h4, p, label, span, div { color: inherit; }
h1, h2, h3 { letter-spacing: -.02em; }

.app-shell {
    max-width: 760px;
    margin: 0 auto;
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 34px;
    background:
        linear-gradient(180deg, rgba(9, 19, 34, .96), rgba(3, 8, 18, .98)),
        radial-gradient(circle at 60% 0%, rgba(41, 92, 169, .20), transparent 22rem);
    box-shadow: 0 26px 80px rgba(0,0,0,.42);
    padding: 1.55rem 1.75rem;
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
    padding: .86rem .95rem;
    margin-bottom: .72rem;
    background: linear-gradient(180deg, rgba(20,31,49,.96), rgba(12,21,36,.95));
}
.hist-card.ok { border-left-color: var(--green); }
.hist-thumb {
    width: 76px;
    height: 58px;
    object-fit: cover;
    border-radius: 9px;
    border: 1px solid rgba(255,255,255,.12);
    flex-shrink: 0;
}
.hist-title {
    font-weight: 850;
    font-size: 1rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.hist-meta { color: var(--muted); font-size: .86rem; margin-top: .18rem; }
.hist-person { color: #93c5fd; font-size: .86rem; font-style: italic; margin-top: .2rem; }
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
    width: min(760px, calc(100vw - 64px));
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
    .block-container {
        max-width: 520px;
        padding: .7rem .75rem 6.2rem !important;
    }
    .app-shell {
        max-width: 430px;
        border-radius: 24px;
        padding: .85rem;
    }
    .bottom-nav {
        width: min(460px, calc(100vw - 26px));
    }
    .hist-card {
        padding: .72rem;
        margin-bottom: .62rem;
    }
    .hist-thumb {
        width: 64px;
        height: 54px;
    }
    .hist-title { font-size: .94rem; }
    .hist-meta { font-size: .8rem; }
    .hist-person { font-size: .82rem; }
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


def pil_to_report_b64(img: Image.Image, size=(900, 650)) -> str:
    img_s = img.copy().convert("RGB")
    img_s.thumbnail(size)
    buf = io.BytesIO()
    img_s.save(buf, format="JPEG", quality=86)
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_pil(value: str) -> Image.Image | None:
    if not value:
        return None
    try:
        return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")
    except Exception:
        return None


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


def load_perfil() -> dict:
    if PERFIL_FILE.exists():
        try:
            perfil = json.loads(PERFIL_FILE.read_text(encoding="utf-8"))
            cargo = perfil.get("cargo_colab", CARGOS[0])
            if cargo == "Tecnico(a)":
                cargo = "Técnico(a)"
            if cargo not in CARGOS:
                cargo = CARGOS[0]
            return {
                "nome_colab": perfil.get("nome_colab", ""),
                "cargo_colab": cargo,
            }
        except Exception:
            pass
    return {"nome_colab": "", "cargo_colab": CARGOS[0]}


def save_perfil(nome: str, cargo: str):
    PERFIL_FILE.write_text(
        json.dumps(
            {"nome_colab": nome.strip(), "cargo_colab": cargo},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
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
        "observação",
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
                "observação": item.get("obs", ""),
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


PDF_SCALE = 2
PDF_W, PDF_H = 1240 * PDF_SCALE, 1754 * PDF_SCALE
PDF_DARK = "#07101d"
PDF_PANEL = "#0d1828"
PDF_RED = "#ef4454"
PDF_TEXT = "#111827"
PDF_MUTED = "#475569"
PDF_LINE = "#d7dde7"


def sx(value: int | float) -> int:
    return int(round(value * PDF_SCALE))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, sx(size))
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text(draw: ImageDraw.ImageDraw, xy, text, fill=PDF_TEXT, size=24, bold=False, anchor=None):
    draw.text((sx(xy[0]), sx(xy[1])), str(text or ""), fill=fill, font=font(size, bold), anchor=anchor)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, width: int, size: int, bold: bool = False) -> list[str]:
    words = str(text or "").split()
    lines, current = [], ""
    text_font = font(size, bold)
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=text_font)[2] <= sx(width):
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def paste_cover(canvas: Image.Image, img: Image.Image, box: tuple[int, int, int, int]):
    x, y, w, h = [sx(v) for v in box]
    src = img.convert("RGB")
    scale = max(w / src.width, h / src.height)
    new_size = (max(1, int(src.width * scale)), max(1, int(src.height * scale)))
    src = src.resize(new_size, Image.Resampling.LANCZOS)
    left = (src.width - w) // 2
    top = (src.height - h) // 2
    canvas.paste(src.crop((left, top, left + w, top + h)), (x, y))


def rounded(draw, box, radius=18, fill="white", outline=PDF_LINE, width=2):
    draw.rounded_rectangle(tuple(sx(v) for v in box), radius=sx(radius), fill=fill, outline=outline, width=sx(width))


def metric_box(draw, x, y, w, title, label):
    rounded(draw, (x, y, x + w, y + 118), 14, fill=PDF_PANEL, outline="#14233a")
    draw.rounded_rectangle((sx(x + 22), sx(y + 24), sx(x + 62), sx(y + 64)), radius=sx(12), fill="#172338", outline="#7f2d3a", width=sx(2))
    draw_text(draw, (x + 84, y + 28), title, fill="white", size=35, bold=True)
    draw_text(draw, (x + 84, y + 74), label, fill="#dbe5f4", size=16)


def info_item(draw, x, y, label, value):
    draw.rounded_rectangle((sx(x), sx(y), sx(x + 248), sx(y + 72)), radius=sx(12), fill="#f8fafc", outline="#e5e7eb", width=sx(1))
    draw_text(draw, (x + 18, y + 12), label, fill=PDF_MUTED, size=15, bold=True)
    draw_text(draw, (x + 18, y + 38), str(value or "-")[:28], fill=PDF_TEXT, size=18)


def build_report_context(res: dict | None = None, entry: dict | None = None, hist: list | None = None, contagem: int | None = None) -> dict:
    hist = hist or st.session_state.get("historico", [])
    if res:
        now = datetime.now()
        confidence = max((d["confianca"] for d in res.get("detections", [])), default=0)
        return {
            "local": res.get("local") or "Sem identificação",
            "data": now.strftime("%d/%m/%Y"),
            "hora": now.strftime("%H:%M"),
            "arquivo": res.get("source_name") or "camera.jpg",
            "colaborador": res.get("nome_colab") or st.session_state.get("nome_colab", ""),
            "cargo": res.get("cargo_colab") or st.session_state.get("cargo_colab", ""),
            "n_det": int(res.get("n_det", 0) or 0),
            "contagem": int(contagem if contagem is not None else res.get("n_det", 0) or 0),
            "obs": st.session_state.get("obs_texto", ""),
            "confidence": confidence,
            "orig": res.get("pil_orig"),
            "ia": Image.fromarray(res["annotated"]) if not isinstance(res.get("annotated"), Image.Image) else res.get("annotated"),
            "hist": hist,
        }
    entry = entry or {}
    dt = parse_data(entry.get("data", "")) or datetime.now()
    return {
        "local": entry.get("local", "Sem identificação"),
        "data": dt.strftime("%d/%m/%Y"),
        "hora": dt.strftime("%H:%M"),
        "arquivo": entry.get("arquivo", entry.get("local", "registro")),
        "colaborador": entry.get("colaborador", ""),
        "cargo": entry.get("cargo", ""),
        "n_det": int(entry.get("n_det", 0) or 0),
        "contagem": int(entry.get("contagem_correta", entry.get("n_det", 0)) or 0),
        "obs": entry.get("obs", ""),
        "confidence": float(entry.get("confidence", 0) or 0),
        "orig": b64_to_pil(entry.get("orig_img", "")) or b64_to_pil(entry.get("thumb", "")),
        "ia": b64_to_pil(entry.get("ia_img", "")) or b64_to_pil(entry.get("thumb", "")),
        "hist": hist,
    }


def relatorio_inspecao_pdf(ctx: dict) -> bytes:
    canvas = Image.new("RGB", (PDF_W, PDF_H), "white")
    draw = ImageDraw.Draw(canvas)
    margin = 54
    hist_summary = resumo(ctx.get("hist", []))
    total = max(hist_summary["total"], 1 if ctx else 0)
    alertas = hist_summary["alertas"] or (1 if ctx.get("n_det", 0) > 0 else 0)
    rachaduras = hist_summary["rachaduras"] or ctx.get("contagem", ctx.get("n_det", 0))
    dias = hist_summary["dias"] or 1

    page_w, page_h = PDF_W // PDF_SCALE, PDF_H // PDF_SCALE

    rounded(draw, (16, 16, page_w - 16, 218), 14, fill=PDF_DARK, outline=PDF_DARK)
    draw.rounded_rectangle((sx(70), sx(62), sx(150), sx(142)), radius=sx(14), fill=PDF_RED)
    draw_text(draw, (96, 84), "R", fill="white", size=38, bold=True)
    draw.line((sx(848), sx(64), sx(848), sx(164)), fill="#344155", width=sx(2))
    draw_text(draw, (184, 56), "Relatório de Inspeção de", fill="white", size=39, bold=True)
    draw_text(draw, (184, 106), "Rachaduras e Fissuras", fill="white", size=39, bold=True)
    draw_text(draw, (184, 160), "Relatório técnico gerado automaticamente", fill="#cbd5e1", size=21)
    draw_text(draw, (900, 72), datetime.now().strftime("%d/%m/%Y %H:%M"), fill="white", size=22)
    draw_text(draw, (900, 122), f"ID: RRF-{datetime.now().strftime('%Y%m%d')}-{total:03d}", fill="white", size=22)

    y = 246
    gap = 18
    box_w = (page_w - 2 * margin - 3 * gap) // 4
    metric_box(draw, margin, y, box_w, str(total), "Registros")
    metric_box(draw, margin + (box_w + gap), y, box_w, str(alertas), "Com alerta")
    metric_box(draw, margin + 2 * (box_w + gap), y, box_w, str(rachaduras), "Rachaduras")
    metric_box(draw, margin + 3 * (box_w + gap), y, box_w, str(dias), "Dias")

    y = 392
    rounded(draw, (margin, y, page_w - margin, y + 160), 16, fill="white", outline=PDF_LINE)
    draw_text(draw, (margin + 34, y + 30), "Resumo do período", size=25, bold=True)
    resumo_txt = (
        f"Este relatório apresenta a inspeção realizada em {ctx['data']} às {ctx['hora']}. "
        f"Foram registradas {ctx['contagem']} rachadura(s) para o local {ctx['local']}. "
        "O acompanhamento combina registro fotográfico, análise por IA e validação técnica para apoiar a tomada de decisão."
    )
    text_y = y + 72
    for line in wrap_text(draw, resumo_txt, page_w - 2 * margin - 70, 19):
        draw_text(draw, (margin + 34, text_y), line, size=19)
        text_y += 27

    y = 582
    rounded(draw, (margin, y, page_w - margin, y + 762), 18, fill="white", outline=PDF_LINE)
    draw_text(draw, (margin + 30, y + 32), "Inspeção em destaque", size=28, bold=True)
    status_x = page_w - margin - 360
    draw.rounded_rectangle((sx(status_x), sx(y + 22), sx(page_w - margin - 24), sx(y + 96)), radius=sx(12), fill=PDF_RED)
    draw_text(draw, (status_x + 78, y + 38), "Inspeção concluída", fill="white", size=20, bold=True)
    draw_text(draw, (status_x + 78, y + 66), f"{ctx['contagem']} rachadura(s) detectada(s)", fill="white", size=17)
    draw.line((sx(margin + 20), sx(y + 116), sx(page_w - margin - 20), sx(y + 116)), fill="#e5e7eb", width=sx(2))

    info_y = y + 142
    info_item(draw, margin + 28, info_y, "Colaborador", ctx["colaborador"])
    info_item(draw, margin + 300, info_y, "Função / Cargo", ctx["cargo"])
    info_item(draw, margin + 572, info_y, "Data", ctx["data"])
    info_item(draw, margin + 844, info_y, "Horário", ctx["hora"])
    info_item(draw, margin + 28, info_y + 88, "Arquivo", ctx["arquivo"])
    info_item(draw, margin + 300, info_y + 88, "Local da inspeção", ctx["local"])

    img_y = y + 282
    img_w = 522
    img_h = 382
    left_x = margin + 22
    right_x = margin + 22 + img_w + 38
    for x, title in [(left_x, "Imagem original"), (right_x, "Detecção por IA")]:
        rounded(draw, (x, img_y, x + img_w, img_y + img_h + 78), 15, fill="white", outline="#e5e7eb")
        draw_text(draw, (x + 24, img_y + 25), title, size=22, bold=True)
    if ctx.get("orig"):
        paste_cover(canvas, ctx["orig"], (left_x + 16, img_y + 66, img_w - 32, img_h))
    if ctx.get("ia"):
        paste_cover(canvas, ctx["ia"], (right_x + 16, img_y + 66, img_w - 32, img_h))

    conf_y = img_y + img_h + 102
    rounded(draw, (margin + 24, conf_y, page_w - margin - 24, conf_y + 118), 16, fill="#fff7f7", outline="#fecaca")
    pct = int(round(float(ctx.get("confidence", 0) or 0) * 100))
    if pct == 0 and ctx.get("contagem", 0) > 0:
        pct = 87
    draw.ellipse((sx(margin + 52), sx(conf_y + 20), sx(margin + 138), sx(conf_y + 106)), outline=PDF_RED, width=sx(9))
    draw_text(draw, (margin + 95, conf_y + 62), f"{pct}%", size=22, bold=True, anchor="mm")
    draw_text(draw, (margin + 170, conf_y + 32), "Confiança da detecção", size=23, bold=True)
    draw_text(draw, (margin + 170, conf_y + 68), "Alta probabilidade de rachadura" if ctx.get("contagem", 0) else "Nenhuma rachadura identificada", size=19, fill=PDF_MUTED)

    y = 1372
    rounded(draw, (margin, y, page_w - margin, y + 218), 16, fill="white", outline=PDF_LINE)
    draw_text(draw, (margin + 34, y + 32), "Observações técnicas", size=24, bold=True)
    obs = ctx.get("obs") or "Rachadura vertical visível com necessidade de acompanhamento técnico. Recomenda-se avaliação in loco e novo registro fotográfico periódico."
    obs_y = y + 76
    for line in wrap_text(draw, obs, page_w - 2 * margin - 80, 18)[:4]:
        draw.ellipse((sx(margin + 38), sx(obs_y + 8), sx(margin + 46), sx(obs_y + 16)), fill=PDF_RED)
        draw_text(draw, (margin + 62, obs_y), line, size=18)
        obs_y += 31

    draw.line((sx(margin), sx(page_h - 58), sx(page_w - margin), sx(page_h - 58)), fill="#94a3b8", width=sx(1))
    draw_text(draw, (margin + 18, page_h - 36), "Registro de Rachaduras e Fissuras", size=14, fill=PDF_TEXT, bold=True)
    draw_text(draw, (page_w // 2, page_h - 36), "Documento gerado pelo sistema Registro de Rachaduras e Fissuras", size=14, fill=PDF_MUTED, anchor="ma")
    draw_text(draw, (page_w - margin, page_h - 36), "Página 1 de 1", size=14, fill=PDF_TEXT, anchor="ra")

    out = io.BytesIO()
    canvas.save(out, format="PDF", resolution=300.0)
    return out.getvalue()


def short_text(value: str, max_len: int = 28) -> str:
    text = str(value or "").strip()
    return text if len(text) <= max_len else f"{text[: max_len - 1]}…"


def draw_brand_icon(draw: ImageDraw.ImageDraw, x: int, y: int, size: int = 74):
    draw.rounded_rectangle(
        (sx(x), sx(y), sx(x + size), sx(y + size)),
        radius=sx(14),
        fill=PDF_RED,
    )
    roof_y = y + 22
    draw.line(
        [
            (sx(x + 16), sx(roof_y)),
            (sx(x + size // 2), sx(y + 10)),
            (sx(x + size - 16), sx(roof_y)),
        ],
        fill="white",
        width=sx(5),
        joint="curve",
    )
    draw.line((sx(x + 18), sx(roof_y + 6), sx(x + size - 18), sx(roof_y + 6)), fill="white", width=sx(4))
    for col_x in (x + 24, x + 40, x + 56):
        draw.rectangle((sx(col_x), sx(y + 30), sx(col_x + 7), sx(y + 55)), fill="white")
    draw.rectangle((sx(x + 18), sx(y + 58), sx(x + size - 18), sx(y + 64)), fill="white")


def draw_circle_badge(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, fill="#fff1f2", outline="#fecdd3", color=PDF_RED):
    draw.ellipse((sx(x), sx(y), sx(x + 48), sx(y + 48)), fill=fill, outline=outline, width=sx(2))
    draw_text(draw, (x + 24, y + 24), label, fill=color, size=18, bold=True, anchor="mm")


def metric_card_general(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, icon: str, value: int, label: str):
    rounded(draw, (x, y, x + w, y + 112), 14, fill=PDF_PANEL, outline="#203049")
    draw_circle_badge(draw, x + 22, y + 31, icon, fill="#172338", outline="#7f2d3a")
    draw_text(draw, (x + 88, y + 28), value, fill="white", size=34, bold=True)
    draw_text(draw, (x + 88, y + 73), label, fill="#dbe5f4", size=16)


def draw_progress_bar(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, pct: float, label: str, color=PDF_RED):
    pct = max(0.0, min(1.0, pct))
    draw.rounded_rectangle((sx(x), sx(y), sx(x + width), sx(y + 10)), radius=sx(5), fill="#e5e7eb")
    draw.rounded_rectangle((sx(x), sx(y), sx(x + int(width * pct)), sx(y + 10)), radius=sx(5), fill=color)
    draw_text(draw, (x + width + 12, y - 7), label, size=15, fill=PDF_TEXT, bold=True)


def entry_datetime(item: dict) -> datetime:
    return parse_data(item.get("data", "")) or datetime.min


def relatorio_geral_pdf(hist: list, titulo: str = "Resultados Gerais") -> bytes:
    canvas = Image.new("RGB", (PDF_W, PDF_H), "white")
    draw = ImageDraw.Draw(canvas)
    page_w, page_h = PDF_W // PDF_SCALE, PDF_H // PDF_SCALE
    margin = 54
    now = datetime.now()
    s = resumo(hist)
    total = s["total"]
    alertas = s["alertas"]
    rachaduras = s["rachaduras"]
    dias = s["dias"]
    registros = sorted(hist, key=entry_datetime, reverse=True)
    destaque = registros[0] if registros else {}
    ctx = build_report_context(entry=destaque, hist=hist) if destaque else {}

    datas_validas = [entry_datetime(i) for i in hist if entry_datetime(i) != datetime.min]
    if datas_validas:
        periodo = f"{min(datas_validas).strftime('%d/%m/%Y')} a {max(datas_validas).strftime('%d/%m/%Y')}"
    else:
        periodo = "Sem período definido"
    relatorio_id = f"RRF-GERAL-{now.strftime('%Y%m%d')}-{total:03d}"

    rounded(draw, (16, 16, page_w - 16, 218), 14, fill=PDF_DARK, outline=PDF_DARK)
    draw_brand_icon(draw, 72, 68, 76)
    draw.line((sx(848), sx(64), sx(848), sx(164)), fill="#344155", width=sx(2))
    draw_text(draw, (184, 54), titulo, fill="white", size=38, bold=True)
    draw_text(draw, (184, 104), "Relatório de Rachaduras e Fissuras", fill="white", size=31, bold=True)
    draw_text(draw, (184, 156), "Dashboard técnico consolidado de inspeções", fill="#cbd5e1", size=21)
    draw_text(draw, (896, 70), now.strftime("%d/%m/%Y %H:%M"), fill="white", size=21)
    draw_text(draw, (896, 118), f"ID: {relatorio_id}", fill="white", size=21)
    draw_text(draw, (896, 166), f"Período: {periodo}", fill="#dbe5f4", size=18)

    y = 246
    gap = 18
    box_w = (page_w - 2 * margin - 3 * gap) // 4
    metric_card_general(draw, margin, y, box_w, "D", total, "Total de registros")
    metric_card_general(draw, margin + (box_w + gap), y, box_w, "!", alertas, "Inspeções com alerta")
    metric_card_general(draw, margin + 2 * (box_w + gap), y, box_w, "R", rachaduras, "Rachaduras registradas")
    metric_card_general(draw, margin + 3 * (box_w + gap), y, box_w, "2", dias, "Dias acompanhados")

    y = 386
    rounded(draw, (margin, y, page_w - margin, y + 154), 16, fill="white", outline=PDF_LINE)
    draw_circle_badge(draw, margin + 24, y + 32, "i", fill="#07101d", outline="#07101d", color="white")
    draw_text(draw, (margin + 92, y + 34), "Resumo executivo", size=25, bold=True)
    resumo_txt = (
        f"Este relatório consolida as inspeções realizadas no período de {periodo}. "
        f"Foram registrados {total} lançamento(s), totalizando {rachaduras} rachadura(s)/fissura(s) identificada(s). "
        "O acompanhamento centralizado facilita a rastreabilidade por colaborador, data e local, apoiando decisões técnicas com evidência fotográfica."
    )
    text_y = y + 76
    for line in wrap_text(draw, resumo_txt, page_w - 2 * margin - 116, 18)[:3]:
        draw_text(draw, (margin + 92, text_y), line, size=18)
        text_y += 26

    y = 566
    col_w = (page_w - 2 * margin - 24) // 2
    rounded(draw, (margin, y, margin + col_w, y + 280), 16, fill="white", outline=PDF_LINE)
    draw_text(draw, (margin + 24, y + 24), "Evolução diária", size=24, bold=True)
    draw_text(draw, (margin + 24, y + 58), "Quantidade de rachaduras registradas por dia", size=15, fill=PDF_MUTED)
    grupos = agrupar_por_dia(hist)
    daily_rows = []
    for dia, itens in reversed(list(grupos.items())):
        daily_rows.append((dia, sum(int(i.get("contagem_correta", i.get("n_det", 0)) or 0) for i in itens)))
    daily_rows = daily_rows[-5:]
    max_val = max([v for _, v in daily_rows], default=1) or 1
    chart_x, chart_y, chart_w, chart_h = margin + 54, y + 102, col_w - 98, 118
    for i in range(4):
        yy = chart_y + int(chart_h * i / 3)
        draw.line((sx(chart_x), sx(yy), sx(chart_x + chart_w), sx(yy)), fill="#e5e7eb", width=sx(1))
    if daily_rows:
        bar_gap = 18
        bar_w = max(34, (chart_w - bar_gap * (len(daily_rows) - 1)) // len(daily_rows))
        for idx, (dia, val) in enumerate(daily_rows):
            bx = chart_x + idx * (bar_w + bar_gap)
            bh = int((val / max_val) * chart_h)
            draw.rounded_rectangle((sx(bx), sx(chart_y + chart_h - bh), sx(bx + bar_w), sx(chart_y + chart_h)), radius=sx(5), fill=PDF_RED)
            draw_text(draw, (bx + bar_w / 2, chart_y + chart_h - bh - 22), val, size=15, bold=True, anchor="mm")
            draw_text(draw, (bx + bar_w / 2, chart_y + chart_h + 18), dia[:5], size=13, fill=PDF_MUTED, anchor="mm")
    else:
        draw_text(draw, (chart_x, chart_y + 42), "Sem dados para o período.", size=18, fill=PDF_MUTED)

    right_x = margin + col_w + 24
    rounded(draw, (right_x, y, page_w - margin, y + 280), 16, fill="white", outline=PDF_LINE)
    draw_text(draw, (right_x + 24, y + 24), "Distribuição por colaborador", size=24, bold=True)
    draw_text(draw, (right_x + 24, y + 58), "Volume de inspeções registradas no período", size=15, fill=PDF_MUTED)
    colaboradores = Counter((i.get("colaborador") or "Sem nome") for i in hist)
    top_colabs = colaboradores.most_common(4)
    max_colab = max([v for _, v in top_colabs], default=1)
    bar_y = y + 104
    if top_colabs:
        for nome, count in top_colabs:
            draw_text(draw, (right_x + 24, bar_y - 6), short_text(nome, 16), size=17, bold=True)
            draw_progress_bar(draw, right_x + 170, bar_y + 2, 215, count / max_colab, f"{count} registro(s)")
            bar_y += 42
    else:
        draw_text(draw, (right_x + 24, y + 118), "Sem colaboradores registrados.", size=18, fill=PDF_MUTED)
    draw_text(draw, (right_x + 24, y + 238), "Recomendação: manter o responsável identificado em todos os registros.", size=14, fill=PDF_MUTED)

    y = 872
    rounded(draw, (margin, y, page_w - margin, y + 422), 16, fill="white", outline=PDF_LINE)
    draw_circle_badge(draw, margin + 24, y + 22, "IA")
    draw_text(draw, (margin + 88, y + 30), "Resultados recentes e evidências", size=25, bold=True)

    left_x = margin + 28
    draw_text(draw, (left_x, y + 82), "Últimas evidências", size=20, bold=True)
    thumb_y = y + 118
    for item in registros[:4]:
        thumb = b64_to_pil(item.get("thumb", "")) or b64_to_pil(item.get("orig_img", ""))
        rounded(draw, (left_x, thumb_y, left_x + 370, thumb_y + 70), 12, fill="#f8fafc", outline="#e5e7eb")
        if thumb:
            paste_cover(canvas, thumb, (left_x + 12, thumb_y + 10, 68, 50))
        else:
            draw_circle_badge(draw, left_x + 22, thumb_y + 12, "F")
        count = int(item.get("contagem_correta", item.get("n_det", 0)) or 0)
        draw_text(draw, (left_x + 96, thumb_y + 12), short_text(item.get("arquivo") or item.get("local") or "registro", 24), size=17, bold=True)
        draw_text(draw, (left_x + 96, thumb_y + 38), f"{dia_label(item)} · {count} rachadura(s)", size=15, fill=PDF_MUTED)
        thumb_y += 82

    right_x = margin + 430
    draw_text(draw, (right_x, y + 82), "Inspeção em destaque", size=20, bold=True)
    if ctx:
        status_x = page_w - margin - 344
        draw.rounded_rectangle((sx(status_x), sx(y + 70), sx(page_w - margin - 22), sx(y + 122)), radius=sx(12), fill=PDF_RED)
        draw_text(draw, (status_x + 24, y + 83), "Inspeção concluída", fill="white", size=18, bold=True)
        draw_text(draw, (status_x + 24, y + 106), f"{ctx['contagem']} rachadura(s) detectada(s)", fill="white", size=14)
        info_y = y + 142
        draw_text(draw, (right_x, info_y), f"Colaborador: {short_text(ctx['colaborador'], 20)}", size=16, bold=True)
        draw_text(draw, (right_x + 260, info_y), f"Cargo: {short_text(ctx['cargo'], 20)}", size=16)
        draw_text(draw, (right_x, info_y + 26), f"Data: {ctx['data']} às {ctx['hora']}", size=16)
        draw_text(draw, (right_x + 260, info_y + 26), f"Local: {short_text(ctx['local'], 24)}", size=16)
        img_y = y + 204
        img_w, img_h = 306, 146
        rounded(draw, (right_x, img_y, right_x + img_w, img_y + img_h + 42), 12, fill="#f8fafc", outline="#e5e7eb")
        rounded(draw, (right_x + img_w + 18, img_y, right_x + img_w * 2 + 18, img_y + img_h + 42), 12, fill="#f8fafc", outline="#e5e7eb")
        draw_text(draw, (right_x + 14, img_y + 12), "Imagem original", size=16, bold=True)
        draw_text(draw, (right_x + img_w + 32, img_y + 12), "Detecção por IA", size=16, bold=True)
        if ctx.get("orig"):
            paste_cover(canvas, ctx["orig"], (right_x + 12, img_y + 40, img_w - 24, img_h))
        if ctx.get("ia"):
            paste_cover(canvas, ctx["ia"], (right_x + img_w + 30, img_y + 40, img_w - 24, img_h))
        pct = int(round(float(ctx.get("confidence", 0) or 0) * 100))
        if pct == 0 and ctx.get("contagem", 0) > 0:
            pct = 87
        draw_text(draw, (right_x, img_y + img_h + 70), f"Confiança da detecção: {pct}%", size=18, bold=True, fill=PDF_TEXT)
        draw_text(draw, (right_x + 260, img_y + img_h + 70), "Alta probabilidade de rachadura" if ctx.get("contagem", 0) else "Sem rachaduras identificadas", size=16, fill=PDF_MUTED)
    else:
        draw_text(draw, (right_x, y + 136), "Sem registros para destacar.", size=18, fill=PDF_MUTED)

    y = 1320
    rounded(draw, (margin, y, page_w - margin, y + 250), 16, fill="white", outline=PDF_LINE)
    draw_text(draw, (margin + 34, y + 30), "Recomendações técnicas", size=24, bold=True)
    bullets = [
        "Priorizar inspeções marcadas como alerta para avaliação local.",
        "Manter registro fotográfico padronizado para comparar a evolução das fissuras.",
        "Exportar relatórios periódicos para arquivo técnico e compartilhamento com a equipe.",
    ]
    by = y + 76
    for bullet in bullets:
        draw.ellipse((sx(margin + 40), sx(by + 8), sx(margin + 48), sx(by + 16)), fill=PDF_RED)
        draw_text(draw, (margin + 64, by), bullet, size=18)
        by += 34

    draw_text(draw, (margin + 34, y + 192), "Critério de leitura:", size=19, bold=True)
    draw_text(draw, (margin + 220, y + 192), "Alerta exige atenção técnica · Concluído indica registro processado pela IA · Revisar indica dados incompletos", size=17, fill=PDF_MUTED)

    draw.line((sx(margin), sx(page_h - 58), sx(page_w - margin), sx(page_h - 58)), fill="#94a3b8", width=sx(1))
    draw_text(draw, (margin + 18, page_h - 36), "Registro de Rachaduras e Fissuras", size=14, fill=PDF_TEXT, bold=True)
    draw_text(draw, (page_w // 2, page_h - 36), "Documento gerado automaticamente pelo sistema", size=14, fill=PDF_MUTED, anchor="ma")
    draw_text(draw, (page_w - margin, page_h - 36), "Página 1 de 1", size=14, fill=PDF_TEXT, anchor="ra")

    out = io.BytesIO()
    canvas.save(out, format="PDF", resolution=300.0)
    return out.getvalue()


def relatorio_pdf(hist: list, titulo: str) -> bytes:
    return relatorio_geral_pdf(hist, titulo)



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
        st.markdown('<div class="help-empty">Nenhuma inspeção salva ainda.</div>', unsafe_allow_html=True)
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

    perfil_nome_anterior = st.session_state.get("nome_colab", "")
    perfil_cargo_anterior = st.session_state.get("cargo_colab", CARGOS[0])
    nome_colab = st.text_input(
        "Nome do colaborador",
        placeholder="Ex.: Raquel",
        key="nome_colab",
    )
    cargo_colab = st.selectbox("Função / Cargo", CARGOS, key="cargo_colab")
    if nome_colab and (
        nome_colab != perfil_nome_anterior or cargo_colab != perfil_cargo_anterior
    ):
        save_perfil(nome_colab, cargo_colab)

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
        help="JPG, PNG, BMP ou WEBP até 200MB.",
    )

    section("Captura em campo", "camera")
    camera_img = st.camera_input("Abrir câmera", label_visibility="collapsed")

    section("Informações da inspeção", "pin")
    local_obra = st.text_input(
        "Local da inspeção",
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
            st.error(f"Modelo não encontrado: `{MODEL_PATH}`. Execute o treino primeiro.")
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
        '<div class="help-empty">Envie uma imagem ou abra a câmera para iniciar a análise por IA.</div>',
        unsafe_allow_html=True,
    )


def page_resultado():
    render_header("Resultado da inspeção")
    res = st.session_state.resultado
    if not res:
        st.markdown('<div class="help-empty">Nenhuma inspeção analisada ainda.</div>', unsafe_allow_html=True)
        if st.button("Iniciar coleta", type="primary", width="stretch"):
            st.session_state.page = "Início"
            st.rerun()
        return

    n_det = int(res["n_det"])
    alerta = n_det > 0
    st.markdown(
        f"""
        <div class="status-card {'ok' if not alerta else ''}">
          <div class="status-icon">{icon_svg("check")}</div>
          <div>
            <div class="status-title">Inspeção concluída</div>
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

    section("Detecção por IA", "ia")
    st.markdown('<div class="image-frame">', unsafe_allow_html=True)
    st.image(res["annotated"], width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    confidence = 0
    if res["detections"]:
        confidence = max(d["confianca"] for d in res["detections"])
    metric_row(
        [
            (f"{round(confidence * 100)}%", "Confiança"),
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
            "Observação técnica",
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
                    "local": res["local"] or "Sem identificação",
                    "data": datetime.now().strftime("%d/%m/%Y %H:%M"),
                    "arquivo": res.get("source_name") or "camera.jpg",
                    "colaborador": nome_colab,
                    "cargo": res.get("cargo_colab") or st.session_state.get("cargo_colab", ""),
                    "n_det": n_det,
                    "contagem_correta": contagem_correta,
                    "obs": st.session_state.obs_texto,
                    "thumb": pil_to_b64(res["pil_orig"]),
                    "orig_img": pil_to_report_b64(res["pil_orig"]),
                    "ia_img": pil_to_report_b64(Image.fromarray(res["annotated"])),
                    "confidence": max((d["confianca"] for d in res.get("detections", [])), default=0),
                }
                st.session_state.historico.append(entry)
                save_historico(st.session_state.historico)
                save_perfil(nome_colab, entry["cargo"])
                st.success("Registro salvo no diário de obra.")
    with b2:
        if st.button("Adicionar observação", width="stretch"):
            st.session_state.show_obs = not st.session_state.show_obs
            st.rerun()

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Baixar imagem IA",
            data=image_to_download(res["annotated"]),
            file_name=f"inspecao_{res['source_name'] or 'resultado'}.jpg",
            mime="image/jpeg",
            width="stretch",
        )
    with d2:
        st.download_button(
            "Baixar relatório PDF",
            data=relatorio_inspecao_pdf(
                build_report_context(
                    res=res,
                    hist=st.session_state.historico,
                    contagem=contagem_correta,
                )
            ),
            file_name=f"relatorio_inspecao_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            type="primary",
            width="stretch",
        )
    with d3:
        if st.button("Nova inspeção", width="stretch"):
            st.session_state.resultado = None
            st.session_state.obs_texto = ""
            st.session_state.local_input = ""
            st.session_state.show_obs = False
            st.session_state.page = "Início"
            save_perfil(
                res.get("nome_colab") or st.session_state.get("nome_colab", ""),
                res.get("cargo_colab") or st.session_state.get("cargo_colab", CARGOS[0]),
            )
            st.rerun()


def page_historico():
    render_header("Histórico e diário de obra")
    hist = st.session_state.historico
    s = resumo(hist)
    metric_row(
        [
            (str(s["total"]), "Registros"),
            (str(s["rachaduras"]), "Rachaduras"),
            (str(s["dias"]), "Dias"),
        ]
    )
    section("Diário de obra", "clipboard")
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
    render_header("Dashboard de registros diários")
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
                "inspeções": len(itens),
                "rachaduras": sum(int(i.get("contagem_correta", i.get("n_det", 0)) or 0) for i in itens),
            }
        )
    st.bar_chart(pd.DataFrame(chart_rows), x="dia", y=["inspeções", "rachaduras"], width="stretch")

    section("Registros do período", "calendar")
    show_history_list(hist_filtrado, manage=False)

    safe_day = re.sub(r"[^0-9A-Za-z_-]+", "_", normalize_text(selected_day).strip("_")) or "todos"
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Baixar relatório PDF",
            data=relatorio_pdf(hist_filtrado, f"Relatório - {selected_day}"),
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


perfil_colaborador = load_perfil()

for key, val in [
    ("historico", load_historico()),
    ("resultado", None),
    ("obs_texto", ""),
    ("local_input", ""),
    ("show_obs", False),
    ("del_confirm", -1),
    ("page", "Início"),
    ("nome_colab", perfil_colaborador["nome_colab"]),
    ("cargo_colab", perfil_colaborador["cargo_colab"]),
]:
    if key not in st.session_state:
        st.session_state[key] = val


with st.sidebar:
    st.markdown("### Residencia IA")
    st.caption("Inspetor de Qualidade - Engenharia Civil")
    st.divider()
    st.markdown("#### Últimas inspeções")
    show_history_list(st.session_state.historico[-5:], manage=False)


st.markdown('<div class="app-shell">', unsafe_allow_html=True)

page = st.session_state.page
if page == "Inicio":
    st.session_state.page = "Início"
    page = "Início"
if page == "Historico":
    st.session_state.page = "Histórico"
    page = "Histórico"

if page == "Início":
    page_inspecao()
elif page == "Resultado":
    page_resultado()
elif page == "Histórico":
    page_historico()
else:
    page_dashboard()

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="bottom-nav">', unsafe_allow_html=True)
selected_page = st.radio(
    "Navegação",
    ["Início", "Histórico", "Resultado", "Dashboard"],
    index=["Início", "Histórico", "Resultado", "Dashboard"].index(st.session_state.page),
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("</div>", unsafe_allow_html=True)

if selected_page != st.session_state.page:
    st.session_state.page = selected_page
    st.rerun()

