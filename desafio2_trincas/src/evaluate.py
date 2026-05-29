"""
Avalia o modelo treinado no conjunto de teste e exporta metricas e visualizacoes.

Uso:
    python src/evaluate.py \
        --weights  models/best_crack_seg_yolo11n.pt \
        --data     data/dataset_split/data.yaml \
        --split    test \
        --conf     0.25 \
        --iou      0.5 \
        --device   0
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Avalia YOLOv11-seg no conjunto de teste")
    p.add_argument("--weights", type=Path,
                   default=Path("models/best_crack_seg_yolo11n.pt"))
    p.add_argument("--data",    type=Path,
                   default=Path("data/dataset_split/data.yaml"))
    p.add_argument("--split",   type=str,  default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--conf",    type=float, default=0.25,
                   help="Threshold de confianca: descarta deteccoes abaixo deste valor. "
                        "0.25 e o padrao YOLO -- ajuste para cima se muitos FP.")
    p.add_argument("--iou",     type=float, default=0.5,
                   help="IoU threshold para NMS e para calculo de TP/FP. "
                        "mAP@0.5 usa exatamente este valor.")
    p.add_argument("--imgsz",   type=int,  default=640)
    p.add_argument("--device",  type=str,  default="0")
    p.add_argument("--save_dir", type=Path, default=Path("results/evaluation"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCarregando modelo: {args.weights}")
    model = YOLO(str(args.weights))

    # Criterios de decisao explicitados conforme requisito do desafio:
    #   conf=0.25: deteccao aceita apenas se confianca >= 25%
    #   iou=0.50:  TP exige IoU >= 50% entre predicao e ground truth (mAP@0.5)
    print(f"\nCriterios: conf>={args.conf}  IoU>={args.iou}  split={args.split}")

    metrics = model.val(
        data    = str(args.data.resolve()),
        split   = args.split,
        conf    = args.conf,
        iou     = args.iou,
        imgsz   = args.imgsz,
        device  = args.device,
        plots   = True,
        save_json = True,
        project = str(args.save_dir),
        name    = f"eval_{args.split}",
    )

    # Extrai e exibe as metricas principais
    box  = metrics.box
    seg  = metrics.seg

    summary = {
        "split": args.split,
        "conf_threshold": args.conf,
        "iou_threshold":  args.iou,
        "box": {
            "mAP50":     float(box.map50),
            "mAP50_95":  float(box.map),
            "precision": float(box.mp),
            "recall":    float(box.mr),
        },
        "seg": {
            "mAP50":     float(seg.map50),
            "mAP50_95":  float(seg.map),
            "precision": float(seg.mp),
            "recall":    float(seg.mr),
        },
    }

    out_json = args.save_dir / f"eval_{args.split}" / "metrics_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Resultados no conjunto de TESTE ===")
    print(f"  {'Metrica':<25} {'BBox':>8}  {'Mask':>8}")
    print(f"  {'-'*44}")
    for k in ["mAP50", "mAP50_95", "precision", "recall"]:
        print(f"  {k:<25} {summary['box'][k]:>8.4f}  {summary['seg'][k]:>8.4f}")

    print(f"\nJSON salvo em: {out_json}")
    print(f"Plots salvo em: {args.save_dir}/eval_{args.split}/")

    # Notas qualitativas impressas como guia de analise
    print("\n=== Guia de analise qualitativa ===")
    print("  - Falsos Positivos comuns: texturas rugosas, sombras lineares,")
    print("    juntas de dilatacao, rebarbas de forma. Verifique nas imagens")
    print("    de inferencia (predict.py) se ha padroes recorrentes.")
    print("  - Falsos Negativos comuns: fissuras muito finas (< ~3 px),")
    print("    trincas com baixo contraste (parede molhada/suja).")
    print("  - Se precision baixa: aumente --conf. Se recall baixo: diminua.")


if __name__ == "__main__":
    main()
