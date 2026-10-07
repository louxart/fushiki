"""
Carga dinámica del modelo propio YOLOv11-cls.

Dos backends con los MISMOS pesos:
  * .pt   -> Ultralytics:   from ultralytics import YOLO; model = YOLO(path)   (requiere PyTorch)
  * .onnx -> onnxruntime:   exportado con  yolo export model=best.pt format=onnx
             (sin PyTorch: es el que se usa en Vercel, donde el bundle máximo es 500 MB)
"""
from __future__ import annotations

import ast
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np


def torch_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("ultralytics") is not None and importlib.util.find_spec("torch") is not None
    except Exception:  # noqa: BLE001
        return False


def _clean(v):
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return str(v)


def training_meta_from_ckpt(ck: dict) -> dict:
    ta = ck.get("train_args") or {}
    keys = ["model", "data", "epochs", "imgsz", "batch", "optimizer", "lr0", "patience", "seed", "augment"]
    tr = ck.get("train_results") or {}
    return {
        "date": str(ck.get("date", "")),
        "ultralytics_version": ck.get("version"),
        "train_args": {k: _clean(ta[k]) for k in keys if k in ta},
        "train_metrics": {k: _clean(v) for k, v in (ck.get("train_metrics") or {}).items()},
        "train_results": {k: _clean(v) for k, v in tr.items()
                          if k in ("epoch", "train/loss", "val/loss", "metrics/accuracy_top1")},
    }


class ModelManager:
    def __init__(self):
        self.model = None          # objeto YOLO (backend pt) o InferenceSession (backend onnx)
        self.backend = None        # "ultralytics" | "onnxruntime"
        self.path = None
        self.names = {}
        self.task = None
        self.imgsz = None
        self.loaded_at = None
        self.load_ms = None
        self.error = None
        self.training = None
        self._lock = threading.RLock()

    @property
    def ready(self):
        return self.model is not None

    # ------------------------------------------------------------------ carga
    def load(self, path: str | Path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No existe el archivo de pesos: {path}")
        t0 = time.perf_counter()
        if path.suffix.lower() == ".onnx":
            self._load_onnx(path)
        else:
            if not torch_available():
                raise RuntimeError(
                    "Este despliegue no tiene PyTorch/Ultralytics (p. ej. Vercel). "
                    "Exporta el modelo a ONNX:  yolo export model=best.pt format=onnx imgsz=96  y sube el .onnx.")
            self._load_pt(path)
        self.path = str(path)
        self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.load_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.error = None
        return self.info()

    def _load_pt(self, path: Path):
        from ultralytics import YOLO, settings  # import diferido (torch es pesado)
        try:
            settings.update({"sync": False})  # sin telemetría
        except Exception:  # noqa: BLE001
            pass
        with self._lock:
            model = YOLO(str(path))
            if model.task != "classify":
                raise ValueError(f"El modelo es de tipo '{model.task}'; se esperaba un clasificador (yolo11n-cls).")
            model.predict(np.full((64, 64, 3), 255, np.uint8), verbose=False)  # calentamiento
            self.model, self.backend, self.task = model, "ultralytics", model.task
            self.names = {int(k): str(v) for k, v in model.names.items()}
            args = getattr(model.model, "args", {}) or {}
            self.imgsz = args.get("imgsz") if isinstance(args, dict) else None
            self.training = training_meta_from_ckpt(getattr(model, "ckpt", None) or {})

    def _load_onnx(self, path: Path):
        import onnxruntime as ort
        with self._lock:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 2
            sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            meta = sess.get_modelmeta().custom_metadata_map
            task = meta.get("task", "classify")
            if task != "classify":
                raise ValueError(f"El modelo ONNX es de tipo '{task}'; se esperaba un clasificador.")
            names = ast.literal_eval(meta["names"]) if "names" in meta else {}
            imgsz = ast.literal_eval(meta["imgsz"]) if "imgsz" in meta else [96, 96]
            self.model, self.backend, self.task = sess, "onnxruntime", task
            self.names = {int(k): str(v) for k, v in names.items()}
            self.imgsz = int(imgsz[0]) if isinstance(imgsz, (list, tuple)) else int(imgsz)
            self._in = sess.get_inputs()[0].name
            # metadatos de entrenamiento exportados junto al .onnx (best.meta.json)
            side = path.with_suffix(".meta.json")
            self.training = json.loads(side.read_text(encoding="utf-8")) if side.exists() else None
            self.classify([np.full((64, 64), 255, np.uint8)])  # calentamiento

    def info(self):
        return {
            "ready": self.ready,
            "backend": self.backend,
            "torch_available": torch_available(),
            "path": self.path,
            "task": self.task,
            "names": self.names,
            "num_classes": len(self.names),
            "imgsz": self.imgsz,
            "loaded_at": self.loaded_at,
            "load_ms": self.load_ms,
            "error": self.error,
            "training": self.training,
        }

    # ------------------------------------------------------------------ inferencia
    def _preprocess_onnx(self, crops):
        """Idéntico a classify_transforms de Ultralytics: BGR->RGB, resize lado corto, center-crop, /255."""
        s = self.imgsz or 96
        batch = []
        for c in crops:
            im = cv2.cvtColor(c, cv2.COLOR_GRAY2RGB) if c.ndim == 2 else cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            h, w = im.shape[:2]
            k = s / min(h, w)
            im = cv2.resize(im, (max(s, round(w * k)), max(s, round(h * k))), interpolation=cv2.INTER_LINEAR)
            h, w = im.shape[:2]
            y0, x0 = (h - s) // 2, (w - s) // 2
            im = im[y0:y0 + s, x0:x0 + s]
            batch.append(im.transpose(2, 0, 1))
        return np.ascontiguousarray(np.stack(batch).astype(np.float32) / 255.0)

    def classify(self, crops_gray):
        """crops_gray: lista de recortes 64x64 (uint8). Devuelve [(label, conf, {label: prob})]."""
        if not self.ready:
            return None
        if self.backend == "onnxruntime":
            with self._lock:
                P = self.model.run(None, {self._in: self._preprocess_onnx(crops_gray)})[0]
        else:
            ims = [cv2.cvtColor(c, cv2.COLOR_GRAY2BGR) if c.ndim == 2 else c for c in crops_gray]
            with self._lock:
                res = self.model.predict(ims, verbose=False, batch=64)
            P = np.stack([r.probs.data.cpu().numpy() for r in res])
        out = []
        for p in P:
            probs = {self.names[i]: float(p[i]) for i in range(len(p))}
            top = int(np.argmax(p))
            out.append((self.names[top], float(p[top]), probs))
        return out
