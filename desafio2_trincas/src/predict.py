"""
Executa inferencia em imagens e salva visualizacoes com mascaras de trincas.

Uso (pasta ou imagem unica):
    python src/predict.py \
        --weights  models/best_crack_seg_yolo11n.pt \
        --source   data/dataset_split/test/images \
        --conf     0.25 \
        --iou      0.5 \
        --save_dir results/predictions
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inferencia YOLOv11-seg em imagens de trincas")
    p.add_argument("--weights",  type=Path,
                   default=Path("models/best_crack_seg_yolo11n.pt"))
    p.add_argument("--source",   type=str,
                   default="data/dataset_split/test/images",
                   help="Caminho para imagem, pasta ou video.")
    p.add_argument("--conf",     type=float, default=0.25)
    p.add_argument("--iou",      type=float, default=0.5)
    p.add_argument("--imgsz",    type=int,   default=640)
    p.add_argument("--device",   type=str,   default="0")
    p.add_argument("--save_dir", type=Path,  default=Path("results/predictions"))
    p.add_argument("--max_det",  type=int,   default=300)
    p.add_argument("--line_width", type=int, default=2,
                   help="Espessura do contorno da mascara na visualizacao.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))

    print(f"\nInferencia em: {args.source}")
    print(f"Conf: {args.conf}  IoU: {args.iou}  Device: {args.device}\n")

    results = model.predict(
        source     = args.source,
        conf       = args.conf,
        iou        = args.iou,
        imgsz      = args.imgsz,
        device     = args.device,
        max_det    = args.max_det,
        save       = True,
        save_conf  = True,
        project    = str(args.save_dir),
        name       = "inference",
        line_width = args.line_width,
        retina_masks = True,   # mascaras em resolucao original (melhor para trincas finas)
    )

    total_detections = sum(len(r.boxes) for r in results)
    print(f"\nTotal de deteccoes: {total_detections}")
    print(f"Imagens salvas em: {args.save_dir}/inference/")


if __name__ == "__main__":
    main()
