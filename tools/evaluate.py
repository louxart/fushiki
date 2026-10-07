"""Evalúa el pipeline sobre la galería de ejemplos contra el ground truth.
Uso: python tools/evaluate.py [motor]   (motor: yolo | hybrid | ocr)"""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.model import ModelManager  # noqa: E402
from pipeline.vision import Detector  # noqa: E402
from pipeline.solver import solve  # noqa: E402

engine = sys.argv[1] if len(sys.argv) > 1 else "hybrid"
mm = ModelManager()
mm.load(ROOT / "weights" / "modelo.pt")
det = Detector(mm)
idx = json.loads((ROOT / "examples" / "index.json").read_text())
for e in idx:
    gt = json.loads((ROOT / "examples" / f"{e['name']}.json").read_text())
    img = cv2.imread(str(ROOT / "examples" / e["image"]))
    try:
        d = det.run(img, engine=engine)
    except Exception as ex:
        print(e["name"], "ERROR", ex); continue
    b, g = d["board"], gt["board"]
    errs = []
    if b["N"] != g["N"]:
        print(e["name"], "N mal", b["N"], g["N"]); continue
    n = g["N"]
    for r in range(n):
        for c in range(n):
            if b["grid"][r][c] != g["grid"][r][c]:
                errs.append(f"cell({r},{c}) {b['grid'][r][c]}!={g['grid'][r][c]}")
    for r in range(n):
        for c in range(n - 1):
            if b["h"][r][c] != g["h"][r][c]:
                errs.append(f"h({r},{c}) '{b['h'][r][c]}'!='{g['h'][r][c]}'")
    for r in range(n - 1):
        for c in range(n):
            if b["v"][r][c] != g["v"][r][c]:
                errs.append(f"v({r},{c}) '{b['v'][r][c]}'!='{g['v'][r][c]}'")
    s = solve(b)
    ok = s["solution"] == gt["solution"]
    print(f"{e['name']:16s} N={n} errores={len(errs)} sol_ok={ok} unica={s['unique']} t={d['timings']['total_vision_ms']}ms", errs[:6])
