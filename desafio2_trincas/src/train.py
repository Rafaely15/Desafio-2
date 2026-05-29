"""
Treina YOLOv11n-seg para deteccao/segmentacao de trincas em paredes.

Uso:
    python src/train.py [--data data/dataset_split/data.yaml]
                        [--epochs 100]
                        [--imgsz 640]
                        [--batch 16]
                        [--device cpu]
                        [--project results/runs]
                        [--name crack_seg_yolo11n]
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


SEED = 42

# Augmentations justificados para dataset pequeno de trincas:
#   - fliplr/flipud: trincas nao tem orientacao canonica
#   - degrees: trincas aparecem em qualquer angulo na parede
#   - hsv_v / hsv_s: variacao de iluminacao e sombras no canteiro
#   - mosaic: aumenta variacao de contexto com dataset pequeno
#   - scale: trincas aparecem em diferentes distancias de captura
#   - shear/perspective: leve deformacao geometrica simula angulos de camera
#   - copy_paste: mistura instancias de segmentacao entre imagens (eficaz para seg)
AUGMENT_OVERRIDES = {
    "fliplr":      0.5,
    "flipud":      0.3,
    "degrees":     15.0,
    "hsv_v":       0.4,
    "hsv_s":       0.4,
    "hsv_h":       0.015,
    "scale":       0.5,
    "shear":       2.0,
    "perspective": 0.0005,
    "mosaic":      1.0,
    "copy_paste":  0.3,
    "mixup":       0.0,   # mixup prejudica bordas finas de trincas
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treina yolo11n-seg em dataset de trincas")
    p.add_argument("--data",    type=Path,
                   default=Path("data/dataset_split/data.yaml"))
    p.add_argument("--model",   type=str,  default="yolo11n-seg.pt",
                   help="Checkpoint de partida (yolo11n-seg.pt para fine-tuning)")
    p.add_argument("--epochs",  type=int,  default=100)
    p.add_argument("--imgsz",   type=int,  default=640,
                   help="Imagens redimensionadas para 640 no treino. "
                        "Originais sao 2560x1440 -- o resize e feito internamente.")
    p.add_argument("--batch",   type=int,  default=16,
                   help="Reduza para 8 se GPU com < 6 GB VRAM, ou use -1 (auto).")
    p.add_argument("--patience",type=int,  default=20,
                   help="Early stopping: para se val/mAP nao melhorar em N epochs.")
    p.add_argument("--device",  type=str,  default="0",
                   help="'0' para GPU CUDA 0, 'cpu' para CPU.")
    p.add_argument("--project", type=str,  default="results/runs")
    p.add_argument("--name",    type=str,  default="crack_seg_yolo11n")
    p.add_argument("--resume",  action="store_true",
                   help="Retoma treino interrompido a partir do last.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.resume:
        last_weights = Path(args.project) / args.name / "weights" / "last.pt"
        if not last_weights.exists():
            raise FileNotFoundError(f"Checkpoint nao encontrado: {last_weights}")
        model = YOLO(str(last_weights))
    else:
        model = YOLO(args.model)

    train_kwargs = dict(
        data      = str(args.data.resolve()),
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        patience  = args.patience,
        device    = args.device,
        project   = args.project,
        name      = args.name,
        seed      = SEED,
        resume    = args.resume,
        # Salvar plots e metricas automaticamente
        plots     = True,
        save      = True,
        val       = True,
        # Evitar que o Ultralytics tente baixar automaticamente outro modelo
        pretrained = True,
        **AUGMENT_OVERRIDES,
    )

    print("\n=== Hiperparametros de treino ===")
    for k, v in train_kwargs.items():
        print(f"  {k:20s}: {v}")
    print()

    results = model.train(**train_kwargs)

    # Copia best.pt para models/ para acesso facil
    best_src = Path(args.project) / args.name / "weights" / "best.pt"
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    if best_src.exists():
        import shutil
        dest = models_dir / "best_crack_seg_yolo11n.pt"
        shutil.copy2(best_src, dest)
        print(f"\nMelhor modelo copiado para: {dest}")

    print(f"\nTreino concluido. Resultados em: {args.project}/{args.name}/")


if __name__ == "__main__":
    main()
