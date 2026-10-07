"""Exporta weights/best.pt a weights/best.onnx (+ best.meta.json con las curvas de entrenamiento).
El .onnx corre con onnxruntime, sin PyTorch: es el que usa el despliegue en Vercel.
Uso:  python tools/export_onnx.py [ruta/al/modelo.pt]"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.model import training_meta_from_ckpt  # noqa: E402

src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "weights" / "modelo.pt"
from ultralytics import YOLO  # noqa: E402

m = YOLO(str(src))
imgsz = (getattr(m.model, "args", {}) or {}).get("imgsz", 96)
out = Path(m.export(format="onnx", imgsz=imgsz, simplify=True, opset=12, dynamic=True))
dst = ROOT / "weights" / (src.stem + ".onnx")
if out.resolve() != dst.resolve():
    shutil.copy(out, dst)
meta = training_meta_from_ckpt(getattr(m, "ckpt", None) or {})
dst.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
print("OK ->", dst)
