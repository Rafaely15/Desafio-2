# Desafio 2 — Detecção de Trincas e Fissuras em Paredes

Sistema de visão computacional baseado em **YOLOv11n-seg** para localizar trincas e fissuras em paredes de construção civil, com suporte a deploy em dispositivos móveis.

## Estrutura do projeto

```
desafio2_trincas/
├── data/dataset_split/        # Dataset split train/val/test (gerado pelo script)
│   ├── train/images | labels
│   ├── val/images   | labels
│   ├── test/images  | labels
│   └── data.yaml
├── src/
│   ├── prepare_dataset.py     # Etapa 1: split + data.yaml
│   ├── train.py               # Etapa 2: treinamento
│   ├── evaluate.py            # Etapa 3: métricas no conjunto de teste
│   ├── predict.py             # Etapa 3: inferência com visualização
│   └── export.py              # Etapa 4: exportação ONNX / TFLite
├── app/
│   └── app.py                 # Etapa 5: interface Streamlit
├── models/                    # Pesos treinados (.pt, .onnx, .tflite)
├── results/                   # Métricas, curvas, imagens de inferência
├── requirements.txt
└── README.md
```

## Pré-requisitos

```bash
# Ambiente conda com PyTorch + CUDA (recomendado)
conda activate yolov11

# Ou instale as dependências
pip install -r requirements.txt
```

> **GPU**: Testado com RTX 4050 (6 GB VRAM), CUDA 12.8.  
> **CPU**: Funciona, mas o treino será muito mais lento.

## Como rodar (passo a passo)

Todos os comandos devem ser executados a partir da raiz `desafio2_trincas/`.

### Etapa 1 — Preparar o dataset

```bash
python src/prepare_dataset.py \
    --src_images ../dataset/images \
    --src_labels ../dataset/labels \
    --dst        data/dataset_split \
    --split      0.70 0.20 0.10 \
    --seed       42
```

Cria `data/dataset_split/` com splits train (70%) / val (20%) / test (10%) e o `data.yaml`.

### Etapa 2 — Treinar

```bash
python src/train.py \
    --data    data/dataset_split/data.yaml \
    --epochs  100 \
    --imgsz   640 \
    --batch   16 \
    --device  0
```

- O melhor modelo é copiado automaticamente para `models/best_crack_seg_yolo11n.pt`.
- Curvas e métricas ficam em `results/runs/crack_seg_yolo11n/`.
- Use `--batch 8` se VRAM < 6 GB.

### Etapa 3 — Avaliar

```bash
# Métricas (mAP, precision, recall) no conjunto de teste
python src/evaluate.py \
    --weights models/best_crack_seg_yolo11n.pt \
    --data    data/dataset_split/data.yaml \
    --split   test \
    --conf    0.25 \
    --iou     0.5

# Gerar imagens de inferência com máscaras
python src/predict.py \
    --weights  models/best_crack_seg_yolo11n.pt \
    --source   data/dataset_split/test/images \
    --save_dir results/predictions
```

### Etapa 4 — Exportar para edge

```bash
python src/export.py \
    --weights models/best_crack_seg_yolo11n.pt \
    --formats onnx tflite
```

Exporta `models/best_crack_seg_yolo11n.onnx` e `.tflite`, e mede latência em CPU.

### Etapa 5 — Interface web

```bash
streamlit run app/app.py
```

Acesse `http://localhost:8501`. Faça upload de uma imagem de parede e veja as trincas detectadas com confiança.

## Decisões de design

| Decisão | Justificativa |
|---|---|
| `yolo11n-seg` em vez de `yolo11n` | Labels já estão em formato de segmentação de polígonos; bounding boxes perderiam precisão para trincas finas e diagonais |
| Split 70/20/10 | 1551 amostras — val com 310 imagens dá estimativa confiável de mAP |
| `imgsz=640` | Equilíbrio entre precisão (trincas finas) e velocidade; originais são 2560×1440 |
| `copy_paste=0.3` | Técnica de augmentation específica para segmentação, aumenta diversidade sem distorcer formas |
| `mixup=0.0` | Mistura de imagens prejudica bordas finas de trincas |
| `patience=20` | Early stopping para evitar overfitting num dataset relativamente pequeno |

## Métricas esperadas

> Valores referência após treinamento completo (veja `results/evaluation/`):

| Métrica | Bbox | Mask |
|---|---|---|
| mAP@0.5 | — | — |
| mAP@0.5:0.95 | — | — |
| Precision | — | — |
| Recall | — | — |

## Notas de deploy mobile

- **ONNX + ONNX Runtime Mobile**: caminho mais direto para Android/iOS
- **TFLite**: ideal para Android com TFLite GPU delegate
- **NCNN**: framework sem dependências, melhor para Android sem Google Play Services
- Modelo nano: ~2.8M parâmetros, ~6 MB `.pt` — viável para dispositivos mid-range
