# Desafio 2 — Detecção de Trincas e Fissuras em Paredes

Sistema de visão computacional para detectar e segmentar trincas e fissuras em paredes de construção civil. O inspetor fotografa a parede, a IA localiza as trincas com máscara colorida e exibe confiança por detecção, com histórico de inspeções e exportação de relatório.

---

## Solução

| Item | Detalhe |
|---|---|
| Modelo | YOLOv11n-seg (segmentação de instâncias) |
| Pesos de produção | `desafio2_trincas/models/best_v3_nano_1280.pt` |
| Interface | Streamlit |
| Porta padrão | 8501 |
| GPU necessária | Não (recomendada para re-treinamento) |
| Tamanho do modelo | ~6 MB |

---

## Estrutura

```
Desafio-2/
├── dataset/                    # Dataset original (imagens + labels)
├── desafio2_trincas/           # Código principal da solução
│   ├── app/app.py              # Interface Streamlit
│   ├── src/                    # Scripts de treino, avaliação e exportação
│   ├── models/                 # Pesos treinados (.pt, .onnx, .tflite)
│   ├── data/dataset_split/     # Dataset dividido (train/val/test)
│   ├── results/                # Métricas, curvas, inferências
│   ├── notebooks/              # Notebooks de análise
│   ├── requirements.txt
│   └── iniciar_app_mobile.bat  # Launcher Windows
├── guia-do-usuario.png
└── guia_ux_ui_registro_rachaduras_fissuras.pdf
```

---

## Como rodar (modelo já treinado)

```bash
cd desafio2_trincas
pip install -r requirements.txt
streamlit run app/app.py
```

Acesse em: `http://localhost:8501`

O Streamlit abrirá o navegador automaticamente. Se não abrir, acesse manualmente.

---

## Como usar o aplicativo

1. Informe **nome** e **cargo** na tela de identificação.
2. Faça **upload** de uma imagem da parede ou use a câmera do dispositivo.
3. A IA detecta as trincas com máscara colorida e exibe a contagem e a confiança de cada detecção.
4. Salve a inspeção para registrá-la no **diário de obra**.
5. Acesse o **Histórico** para ver todas as inspeções salvas e exportar em CSV.
6. No **Dashboard** visualize indicadores por dia e por responsável.

---

## Acesso pelo smartphone

Na mesma rede Wi-Fi:

1. Descubra o IP do computador: `ipconfig` (Windows)
2. Acesse no celular: `http://<SEU_IP>:8501`

Para acesso remoto (fora da rede local):

```bash
cloudflared tunnel --url http://localhost:8501
```

O terminal exibirá uma URL `https://xxxxx.trycloudflare.com` — use essa URL no celular.

---

## Re-treinamento (opcional)

Caso queira treinar novamente com novos dados:

```bash
cd desafio2_trincas

# Etapa 1 — Preparar o dataset
python src/prepare_dataset.py \
    --src_images ../dataset/images \
    --src_labels ../dataset/labels \
    --dst        data/dataset_split \
    --split      0.70 0.20 0.10 \
    --seed       42

# Etapa 2 — Treinar
python src/train.py \
    --data    data/dataset_split/data.yaml \
    --epochs  100 \
    --imgsz   1280 \
    --batch   8 \
    --device  0

# Etapa 3 — Avaliar
python src/evaluate.py \
    --weights models/best_crack_seg_yolo11n.pt \
    --data    data/dataset_split/data.yaml \
    --split   test \
    --conf    0.25

# Etapa 4 — Exportar para dispositivos móveis (opcional)
python src/export.py \
    --weights models/best_crack_seg_yolo11n.pt \
    --formats onnx tflite
```

> Use `--batch 4` se VRAM < 6 GB. Tempo estimado com GPU (RTX 4050): ~2 horas para 100 épocas.

---

## Deploy mobile

| Formato | Uso recomendado |
|---|---|
| `.onnx` | Android / iOS via ONNX Runtime Mobile |
| `.tflite` | Android com TFLite GPU delegate |
| `.ncnn` | Android sem dependências externas |

Modelo nano: ~2.8 M parâmetros, ~6 MB — viável em dispositivos mid-range.

---

## Download

Pasta completa via Google Drive:
[https://drive.google.com/file/d/1M-knoev37IBSK6P4jWJIQSwxHYDUBPFU/view?usp=sharing](https://drive.google.com/file/d/1M-knoev37IBSK6P4jWJIQSwxHYDUBPFU/view?usp=sharing)

Clone via Git:

```bash
git clone https://github.com/Rafaely15/Desafio-2.git
```

---

Documentação detalhada: [desafio2_trincas/README.md](desafio2_trincas/README.md)

Bootcamp CDIA — Programa de Residência em IA — Ana Rafaely — 2026
