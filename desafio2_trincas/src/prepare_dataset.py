"""
Organiza o dataset bruto em splits train/val/test no padrão Ultralytics
e gera o data.yaml para tarefa de segmentação de instâncias (yolo11n-seg).

Uso:
    python src/prepare_dataset.py \
        --src_images  dataset/images \
        --src_labels  dataset/labels \
        --dst          data/dataset_split \
        --split        0.70 0.20 0.10 \
        --seed         42
"""

import argparse
import random
import shutil
from pathlib import Path

import yaml


# ── Proporções padrão ──────────────────────────────────────────────────────────
DEFAULT_SPLIT = (0.70, 0.20, 0.10)
SEED = 42
IMG_EXT = ".jpg"
LBL_EXT = ".txt"
CLASS_NAMES = ["rachadura"]       # única classe identificada na Etapa 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split dataset para YOLOv11-seg")
    p.add_argument("--src_images", type=Path, default=Path("dataset/images"))
    p.add_argument("--src_labels", type=Path, default=Path("dataset/labels"))
    p.add_argument("--dst",        type=Path, default=Path("data/dataset_split"))
    p.add_argument("--split", type=float, nargs=3,
                   default=list(DEFAULT_SPLIT),
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="Proporções que devem somar 1.0")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def validate_split(ratios: list[float]) -> None:
    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split deve somar 1.0, mas soma {total:.4f}")


def collect_paired_stems(src_images: Path, src_labels: Path) -> list[str]:
    """Retorna os stems (nome sem extensão) que têm imagem E label."""
    img_stems = {p.stem for p in src_images.glob(f"*{IMG_EXT}")}
    lbl_stems = {p.stem for p in src_labels.glob(f"*{LBL_EXT}")}
    paired = sorted(img_stems & lbl_stems)

    orphan_imgs = img_stems - lbl_stems
    orphan_lbls = lbl_stems - img_stems
    if orphan_imgs:
        print(f"[AVISO] {len(orphan_imgs)} imagens sem label — ignoradas.")
    if orphan_lbls:
        print(f"[AVISO] {len(orphan_lbls)} labels sem imagem — ignoradas.")

    return paired


def split_stems(stems: list[str], ratios: list[float], seed: int
                ) -> dict[str, list[str]]:
    rng = random.Random(seed)
    shuffled = stems[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    # test recebe o restante para evitar perda de amostras por arredondamento
    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }


def copy_split(stems: dict[str, list[str]],
               src_images: Path, src_labels: Path,
               dst: Path) -> None:
    for split_name, split_stems_ in stems.items():
        img_dst = dst / split_name / "images"
        lbl_dst = dst / split_name / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for stem in split_stems_:
            shutil.copy2(src_images / f"{stem}{IMG_EXT}", img_dst / f"{stem}{IMG_EXT}")
            shutil.copy2(src_labels / f"{stem}{LBL_EXT}", lbl_dst / f"{stem}{LBL_EXT}")

        print(f"  {split_name:5s}: {len(split_stems_):4d} imagens -> {img_dst}")


def write_data_yaml(dst: Path, class_names: list[str]) -> Path:
    # Ultralytics espera paths absolutos ou relativos ao local do yaml
    config = {
        "path": str(dst.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    len(class_names),
        "names": class_names,
    }
    yaml_path = dst / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return yaml_path


def main() -> None:
    args = parse_args()
    validate_split(args.split)

    print(f"\n[1/4] Coletando pares imagem+label...")
    stems = collect_paired_stems(args.src_images, args.src_labels)
    print(f"      {len(stems)} pares encontrados.")

    print(f"\n[2/4] Dividindo com seed={args.seed}, split={args.split}...")
    splits = split_stems(stems, args.split, args.seed)
    for k, v in splits.items():
        print(f"      {k:5s}: {len(v)} ({len(v)/len(stems)*100:.1f}%)")

    print(f"\n[3/4] Copiando arquivos para '{args.dst}'...")
    copy_split(splits, args.src_images, args.src_labels, args.dst)

    print(f"\n[4/4] Gerando data.yaml...")
    yaml_path = write_data_yaml(args.dst, CLASS_NAMES)
    print(f"      Salvo em: {yaml_path}")

    # Verificação rápida de integridade
    for split_name in ["train", "val", "test"]:
        n_imgs = len(list((args.dst / split_name / "images").glob(f"*{IMG_EXT}")))
        n_lbls = len(list((args.dst / split_name / "labels").glob(f"*{LBL_EXT}")))
        status = "OK" if n_imgs == n_lbls else "ERRO - DESBALANCEADO"
        print(f"      [{status}] {split_name}: {n_imgs} imgs / {n_lbls} labels")

    print("\nDone.")


if __name__ == "__main__":
    main()
