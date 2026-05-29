"""
Treina YOLOv11-seg para deteccao/segmentacao de rachaduras em paredes.

Experimentos recomendados (em ordem de impacto esperado):

  Baseline (ja treinado):
    python src/train.py --epochs 100 --imgsz 640  --model yolo11n-seg.pt  --name v1_nano_640

  Exp 1 — mais epocas (modelo nao havia convergido em 100):
    python src/train.py --epochs 200 --imgsz 640  --model yolo11n-seg.pt  --name v2_nano_640_200ep

  Exp 2 — resolucao maior (maior ganho para fissuras finas):
    python src/train.py --epochs 150 --imgsz 1280 --model yolo11n-seg.pt  --batch 8 --name v3_nano_1280

  Exp 3 — modelo maior (mais capacidade):
    python src/train.py --epochs 150 --imgsz 640  --model yolo11s-seg.pt  --name v4_small_640

  Exp 4 — combinacao otima:
    python src/train.py --epochs 150 --imgsz 1280 --model yolo11s-seg.pt  --batch 8 --name v5_small_1280
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


SEED = 42

# Augmentations para dataset pequeno de rachaduras.
# Melhorias em relacao ao v1:
#   - erasing 0.4->0.2: random erasing apagava pixels de fissuras finas, prejudicando recall
#   - copy_paste 0.3->0.5: mais mixtura de instancias melhora generalizacao em dataset pequeno
#   - degrees 15->20: rachaduras podem ter qualquer orientacao na parede
#   - cos_lr=True: learning rate cosine annealing converge melhor em treinos longos (>100 ep)
AUGMENT_OVERRIDES = {
    "fliplr":      0.5,
    "flipud":      0.3,
    "degrees":     20.0,
    "hsv_v":       0.4,
    "hsv_s":       0.4,
    "hsv_h":       0.015,
    "scale":       0.5,
    "shear":       2.0,
    "perspective": 0.0005,
    "mosaic":      1.0,
    "copy_paste":  0.5,   # aumentado: mais diversidade de instancias
    "mixup":       0.0,   # mixup prejudica bordas finas
    "erasing":     0.2,   # reduzido: evitar apagar pixels de fissuras finas
    "cos_lr":      True,  # cosine LR annealing — melhor para treinos longos
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treina yolo11n-seg em dataset de trincas")
    p.add_argument("--data",    type=Path,
                   default=Path("data/dataset_split/data.yaml"))
    p.add_argument("--model",   type=str,  default="yolo11n-seg.pt",
                   help="yolo11n-seg.pt (nano) ou yolo11s-seg.pt (small, +3-5 mAP, +40%% VRAM)")
    p.add_argument("--epochs",  type=int,  default=200,
                   help="200 recomendado: o modelo ainda nao havia convergido em 100 epocas.")
    p.add_argument("--imgsz",   type=int,  default=640,
                   help="Use 1280 para melhor deteccao de fissuras finas "
                        "(originais 2560x1440 -- downscale 4:1 a 640 apaga detalhes).")
    p.add_argument("--batch",   type=int,  default=16,
                   help="Use 8 para imgsz=1280 na RTX 4050 (6 GB VRAM).")
    p.add_argument("--patience",type=int,  default=30,
                   help="Aumentado para 30: evita parar cedo em platôs temporarios.")
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
        # Nome do destino inclui o experimento para nao sobrescrever versoes anteriores
        safe_name = args.name.replace("/", "_")
        dest = models_dir / f"best_{safe_name}.pt"
        shutil.copy2(best_src, dest)
        print(f"\nMelhor modelo copiado para: {dest}")

    print(f"\nTreino concluido. Resultados em: {args.project}/{args.name}/")


if __name__ == "__main__":
    main()
