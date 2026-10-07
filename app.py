"""
Futoshiki Solver - backend FastAPI.

Ejecutar:
    pip install -r requirements.txt
    python app.py              (o: uvicorn app:app --reload)
    -> http://localhost:8000
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline.model import ModelManager, torch_available
from pipeline.solver import normalize_board, solve, to_ascii, to_python_matrix
from pipeline.vision import Detector, b64img, render_projection, render_topview

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights"
EXAMPLES = ROOT / "examples"
ON_VERCEL = bool(os.environ.get("VERCEL"))


def default_weights() -> Path | None:
    """FUTOSHIKI_WEIGHTS > best.pt (si hay PyTorch) > best.onnx (sin PyTorch, p. ej. Vercel)."""
    env = os.environ.get("FUTOSHIKI_WEIGHTS")
    if env:
        return Path(env) if Path(env).is_absolute() else ROOT / env
    cands = (["best.pt", "best.onnx"] if torch_available() else ["best.onnx"])
    for c in cands:
        if (WEIGHTS / c).is_file():
            return WEIGHTS / c
    return None


def upload_dir() -> Path:
    """weights/ si se puede escribir; si no (Vercel: sistema de archivos de solo lectura) /tmp."""
    for d in ([] if ON_VERCEL else [WEIGHTS]) + [Path(tempfile.gettempdir()) / "futoshiki_weights"]:
        try:
            d.mkdir(parents=True, exist_ok=True)
            t = d / ".w"; t.write_text("ok"); t.unlink()
            return d
        except OSError:
            continue
    raise HTTPException(500, "No hay un directorio con permiso de escritura para el modelo.")

app = FastAPI(title="Futoshiki Solver", version="1.0")
mm = ModelManager()
detector = Detector(mm)
CACHE: "OrderedDict[str, dict]" = OrderedDict()   # últimas detecciones (para re-proyectar tras edición manual)


_autoload_lock = threading.Lock()
_autoload_done = False


def ensure_model():
    """Carga perezosa del modelo por defecto (también funciona en serverless, donde no hay 'startup')."""
    global _autoload_done
    if _autoload_done or mm.ready:
        return
    with _autoload_lock:
        if _autoload_done or mm.ready:
            return
        _autoload_done = True
        w = default_weights()
        if w is None:
            mm.error = "No se encontró weights/best.pt ni weights/best.onnx"
            return
        try:
            mm.load(w)
            print(f"[modelo] cargado {w} ({mm.backend})  clases={mm.names}")
        except Exception as e:  # noqa: BLE001
            mm.error = str(e)
            print("[modelo] error:", e)


@app.on_event("startup")
def _autoload():
    if not ON_VERCEL:
        ensure_model()


# ------------------------------------------------------------------------------------ estado
@app.get("/api/status")
def status():
    ensure_model()
    import ortools
    versions = {"ortools": ortools.__version__, "opencv": cv2.__version__}
    try:
        import onnxruntime
        versions["onnxruntime"] = onnxruntime.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import ultralytics, torch  # noqa: E401
        versions.update(ultralytics=ultralytics.__version__, torch=torch.__version__)
    except Exception:  # noqa: BLE001
        pass
    return {"model": mm.info(), "versions": versions, "platform": "vercel" if ON_VERCEL else "local"}


class PathReq(BaseModel):
    path: str


@app.post("/api/model/path")
def model_from_path(req: PathReq):
    p = Path(req.path)
    if not p.is_absolute():
        p = ROOT / p
    try:
        return mm.load(p)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, str(e))


@app.post("/api/model/upload")
async def model_upload(file: UploadFile = File(...)):
    name = Path(file.filename or "modelo.pt").name
    if not name.lower().endswith((".pt", ".zip", ".onnx")):
        raise HTTPException(400, "Sube un archivo de pesos .pt (Ultralytics) u .onnx (exportado).")
    if name.lower().endswith((".pt", ".zip")) and not torch_available():
        raise HTTPException(400, "Este despliegue no incluye PyTorch (límite de 500 MB de Vercel). "
                                 "Exporta tu modelo con: yolo export model=best.pt format=onnx imgsz=96 — y sube el .onnx.")
    stem = name[:-4] if name.lower().endswith(".zip") else name
    dst = upload_dir() / ("uploaded_" + stem)
    if dst.suffix.lower() not in (".pt", ".onnx"):
        dst = dst.with_suffix(".pt")
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        return mm.load(dst)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"No se pudo cargar el modelo: {e}")


# ------------------------------------------------------------------------------------ ejemplos
@app.get("/api/examples")
def examples():
    idx = EXAMPLES / "index.json"
    return json.loads(idx.read_text(encoding="utf-8")) if idx.exists() else []


# ------------------------------------------------------------------------------------ resolver
def _exports(board, solution):
    b = normalize_board(board)
    return {
        "json": {"N": b["N"], "grid": b["grid"], "h": b["h"], "v": b["v"], "solution": solution},
        "python": to_python_matrix(b) + (f"\nsolution = {solution}" if solution else ""),
        "ascii": to_ascii(b),
        "ascii_solution": to_ascii(b, solution) if solution else None,
    }


@app.post("/api/solve-image")
async def solve_image(
    image: UploadFile | None = File(None),
    example: str | None = Form(None),
    engine: str = Form("hybrid"),
    n: str | None = Form(None),
):
    t0 = time.perf_counter()
    if example:
        p = EXAMPLES / Path(example).name
        if not p.is_file():
            raise HTTPException(404, "Ejemplo no encontrado")
        img = cv2.imread(str(p))
        fname = p.name
    elif image is not None:
        data = np.frombuffer(await image.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        fname = image.filename
    else:
        raise HTTPException(400, "Envía una imagen o un ejemplo.")
    if img is None:
        raise HTTPException(400, "No se pudo leer la imagen (usa JPG, PNG o WEBP).")
    ensure_model()
    if engine not in ("yolo", "hybrid", "ocr"):
        engine = "hybrid"
    n_hint = int(n) if n and n.isdigit() else None

    try:
        det = detector.run(img, engine=engine, n_hint=n_hint)
    except ValueError as e:
        raise HTTPException(422, str(e))
    t_vis = (time.perf_counter() - t0) * 1000

    res = solve(det["board"])
    sol = res["solution"] or (res["diagnosis"] or {}).get("repaired_solution")
    proj = render_projection(det, sol)
    top = render_topview(det, sol)

    det_id = uuid.uuid4().hex[:12]
    CACHE[det_id] = det
    while len(CACHE) > 20:
        CACHE.popitem(last=False)

    dbg = det["_dbg"]
    public = {k: v for k, v in det.items() if not k.startswith("_")}
    return JSONResponse({
        "det_id": det_id,
        "filename": fname,
        "detection": public,
        "solve": res,
        "images": {
            "original": b64img(dbg["original"], max_w=1280),
            "projection": b64img(proj, max_w=1280),
            "topview": b64img(top, max_w=900),
            "binaria": b64img(dbg["binaria"], max_w=900),
            "fase1_hoja": b64img(dbg["fase1_hoja"], max_w=900),
            "fase2_celdas": b64img(dbg["fase2_celdas"], max_w=900),
            "segmentacion": b64img(dbg["segmentacion"], max_w=900),
            "mosaic": b64img(det["_mosaic"], quality=92),
        },
        "exports": _exports(det["board"], sol),
        "timing": {"vision_ms": round(t_vis, 1), "total_ms": round((time.perf_counter() - t0) * 1000, 1)},
    })


class BoardReq(BaseModel):
    board: dict
    det_id: str | None = None


@app.post("/api/solve-board")
def solve_board(req: BoardReq):
    t0 = time.perf_counter()
    try:
        b = normalize_board(req.board)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Tablero inválido: {e}")
    res = solve(b)
    sol = res["solution"] or (res["diagnosis"] or {}).get("repaired_solution")
    out = {"solve": res, "exports": _exports(b, sol), "images": None}
    # re-proyección sobre la foto si el tablero proviene de una detección y conserva N
    det = CACHE.get(req.det_id or "")
    if det is not None and det["board"]["N"] == b["N"]:
        d2 = dict(det); d2["board"] = b
        out["images"] = {
            "projection": b64img(render_projection(d2, sol), max_w=1280),
            "topview": b64img(render_topview(d2, sol), max_w=900),
        }
    out["timing"] = {"total_ms": round((time.perf_counter() - t0) * 1000, 1)}
    return out


# ------------------------------------------------------------------------------------ estáticos
app.mount("/examples", StaticFiles(directory=EXAMPLES), name="examples")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)))
