"""
Exporta o modelo treinado para ONNX e TFLite (edge/mobile deployment).

Uso:
    python src/export.py \
        --weights  models/best_crack_seg_yolo11n.pt \
        --formats  onnx tflite \
        --imgsz    640
"""

import argparse
import time
from pathlib import Path

from ultralytics import YOLO


SUPPORTED_FORMATS = ["onnx", "tflite", "ncnn", "torchscript", "coreml"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exporta YOLOv11-seg para formatos edge")
    p.add_argument("--weights", type=Path,
                   default=Path("models/best_crack_seg_yolo11n.pt"))
    p.add_argument("--formats", nargs="+",
                   default=["onnx", "tflite"],
                   choices=SUPPORTED_FORMATS)
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--device",  type=str, default="cpu",
                   help="CPU recomendado para export (mais estavel que GPU).")
    p.add_argument("--int8",    action="store_true",
                   help="Quantizacao INT8 para TFLite (menor tamanho, mais rapido).")
    p.add_argument("--half",    action="store_true",
                   help="FP16 para ONNX (reduz tamanho ~50%%; requer hardware compativel).")
    return p.parse_args()


def benchmark_onnx(model_path: Path, imgsz: int) -> None:
    """Mede latencia media de inferencia ONNX em CPU (10 runs de warmup + 20 medicoes)."""
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(model_path),
                                    providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

        # Warmup
        for _ in range(10):
            sess.run(None, {inp_name: dummy})

        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            sess.run(None, {inp_name: dummy})
            times.append((time.perf_counter() - t0) * 1000)

        avg = sum(times) / len(times)
        print(f"  Latencia ONNX CPU (imgsz={imgsz}): {avg:.1f} ms/frame  "
              f"(~{1000/avg:.0f} FPS)")
    except ImportError:
        print("  [AVISO] onnxruntime nao instalado -- benchmark pulado.")


def main() -> None:
    args = parse_args()
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model = YOLO(str(args.weights))
    print(f"\nModelo base: {args.weights}")
    print(f"Formatos alvo: {args.formats}\n")

    exported_paths = {}

    for fmt in args.formats:
        print(f"--- Exportando para {fmt.upper()} ---")

        extra = {}
        if fmt == "tflite":
            extra["int8"] = args.int8
        if fmt == "onnx":
            extra["half"] = args.half
            extra["simplify"] = True   # remove nos desnecessarios do grafo

        export_path = model.export(
            format = fmt,
            imgsz  = args.imgsz,
            device = args.device,
            **extra,
        )
        exported_paths[fmt] = Path(export_path)

        size_mb = Path(export_path).stat().st_size / 1e6
        print(f"  Salvo em: {export_path}")
        print(f"  Tamanho: {size_mb:.2f} MB")

        # Copia para models/ (apenas se nao estiver ja la)
        import shutil
        dst = models_dir / Path(export_path).name
        if Path(export_path).resolve() != dst.resolve():
            shutil.copy2(export_path, dst)
            print(f"  Copiado para: {dst}")
        else:
            print(f"  Ja em: {dst}")

    if "onnx" in exported_paths:
        print("\n--- Benchmark ONNX em CPU ---")
        benchmark_onnx(exported_paths["onnx"], args.imgsz)

    print("\n=== Resumo de tamanhos ===")
    for fmt, path in exported_paths.items():
        size_mb = path.stat().st_size / 1e6
        print(f"  {fmt.upper():12s}: {size_mb:.2f} MB")

    # Nota sobre formatos alternativos para Android
    print("\n=== Notas sobre deploy em dispositivo movel ===")
    print("  ONNX + ONNX Runtime Mobile: caminho mais simples para Android/iOS.")
    print("  TFLite: ecossistema TensorFlow Lite -- bom suporte Android (TFLite GPU delegate).")
    print("  NCNN: framework da Tencent, zero dependencias, ideal para Android sem Google Play")
    print("        Services. Exige compilacao nativa; use `model.export(format='ncnn')`.")
    print("  CoreML: exclusivo iOS/macOS -- use `model.export(format='coreml')`.")


if __name__ == "__main__":
    main()
