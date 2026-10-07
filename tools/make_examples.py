"""
Genera la galería de casos de prueba: puzzles Futoshiki con solución única,
"impresos" en una hoja y fotografiados sintéticamente con distintos ángulos,
sombras e iluminación. Guarda la foto JPG + el ground truth JSON.

Uso:  python tools/make_examples.py
"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.solver import solve, empty_board  # noqa: E402

OUT = ROOT / "examples"


# ----------------------------------------------------------------- generador de puzzles
def latin_square(n, rng):
    base = [[(r + c) % n + 1 for c in range(n)] for r in range(n)]
    rows = list(range(n)); rng.shuffle(rows)
    cols = list(range(n)); rng.shuffle(cols)
    syms = list(range(1, n + 1)); rng.shuffle(syms)
    return [[syms[base[rows[r]][cols[c]] - 1] for c in range(n)] for r in range(n)]


def make_puzzle(n, rng, ineq_ratio=0.45):
    sol = latin_square(n, rng)
    b = empty_board(n)
    for r in range(n):
        for c in range(n - 1):
            if rng.random() < ineq_ratio:
                b["h"][r][c] = "<" if sol[r][c] < sol[r][c + 1] else ">"
    for r in range(n - 1):
        for c in range(n):
            if rng.random() < ineq_ratio:
                b["v"][r][c] = "^" if sol[r][c] < sol[r + 1][c] else "v"
    cells = [(r, c) for r in range(n) for c in range(n)]
    rng.shuffle(cells)
    for r, c in cells:
        res = solve(b, check_unique=True, diagnose=False)
        if res["unique"]:
            break
        b["grid"][r][c] = sol[r][c]
    # poda: retirar pistas sobrantes manteniendo unicidad
    for r, c in cells:
        if b["grid"][r][c]:
            v = b["grid"][r][c]
            b["grid"][r][c] = 0
            if not solve(b, check_unique=True, diagnose=False)["unique"]:
                b["grid"][r][c] = v
    return b, sol


# ----------------------------------------------------------------- render de la hoja
def draw_sign(img, cx, cy, sign, size, th, color, rng):
    a = size / 2
    j = lambda: rng.uniform(-a * 0.08, a * 0.08)
    if sign == "<":
        pts = [(cx + a * 0.55 + j(), cy - a * 0.75 + j()), (cx - a * 0.55 + j(), cy + j()), (cx + a * 0.55 + j(), cy + a * 0.75 + j())]
    elif sign == ">":
        pts = [(cx - a * 0.55 + j(), cy - a * 0.75 + j()), (cx + a * 0.55 + j(), cy + j()), (cx - a * 0.55 + j(), cy + a * 0.75 + j())]
    elif sign == "^":
        pts = [(cx - a * 0.75 + j(), cy + a * 0.55 + j()), (cx + j(), cy - a * 0.55 + j()), (cx + a * 0.75 + j(), cy + a * 0.55 + j())]
    else:  # v
        pts = [(cx - a * 0.75 + j(), cy - a * 0.55 + j()), (cx + j(), cy + a * 0.55 + j()), (cx + a * 0.75 + j(), cy - a * 0.55 + j())]
    p = np.array(pts, np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [p], False, color, th, cv2.LINE_AA)


def render_sheet(b, rng, S=150, G=78, font=cv2.FONT_HERSHEY_SIMPLEX):
    n = b["N"]
    W = n * S + (n - 1) * G
    mx, my = 170, 260
    sheet = np.full((W + 2 * my, W + 2 * mx, 3), 248, np.uint8)
    ink = (35, 35, 40)
    # título
    cv2.putText(sheet, f"FUTOSHIKI {n}x{n}", (mx, 140), cv2.FONT_HERSHEY_DUPLEX, 2.0, ink, 3, cv2.LINE_AA)
    for r in range(n):
        for c in range(n):
            x0 = mx + c * (S + G); y0 = my + r * (S + G)
            cv2.rectangle(sheet, (x0, y0), (x0 + S, y0 + S), ink, 5, cv2.LINE_AA)
            v = b["grid"][r][c]
            if v:
                txt = str(v)
                scale = S / 38
                (tw, th), _ = cv2.getTextSize(txt, font, scale, 9)
                cv2.putText(sheet, txt, (x0 + (S - tw) // 2, y0 + (S + th) // 2), font, scale, ink, 9, cv2.LINE_AA)
    for r in range(n):
        for c in range(n - 1):
            s = b["h"][r][c]
            if s:
                cx = mx + c * (S + G) + S + G / 2; cy = my + r * (S + G) + S / 2
                draw_sign(sheet, cx, cy, s, G * 0.75, 6, ink, rng)
    for r in range(n - 1):
        for c in range(n):
            s = b["v"][r][c]
            if s:
                cx = mx + c * (S + G) + S / 2; cy = my + r * (S + G) + S + G / 2
                draw_sign(sheet, cx, cy, s, G * 0.75, 6, ink, rng)
    # leve textura de papel
    noise = rng_np(rng).normal(0, 3, sheet.shape[:2])
    sheet = np.clip(sheet.astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)
    return sheet


def rng_np(rng):
    return np.random.default_rng(rng.randint(0, 10**9))


def photograph(sheet, rng, tilt=0.12, rot=0.0, shadow=0.0, dim=1.0, warm=0.0, out_size=(1600, 1200)):
    """Coloca la hoja sobre una mesa y aplica perspectiva, sombra y ruido de cámara."""
    Wo, Ho = out_size
    npr = rng_np(rng)
    # mesa (madera / gris)
    yy, xx = np.mgrid[0:Ho, 0:Wo]
    grain = (np.sin(xx / 9.0 + np.sin(yy / 50.0) * 3) * 10).astype(np.float32)
    base = np.array([70, 95, 125], np.float32) if warm > 0 else np.array([95, 92, 88], np.float32)
    table = base[None, None, :] + grain[..., None] + npr.normal(0, 6, (Ho, Wo, 1))
    table = np.clip(table, 0, 255).astype(np.uint8)

    h, w = sheet.shape[:2]
    scale = min(Wo * 0.72 / w, Ho * 0.82 / h)
    cw, ch = w * scale, h * scale
    cx, cy = Wo / 2 + rng.uniform(-40, 40), Ho / 2 + rng.uniform(-30, 30)
    quad = np.array([[-cw / 2, -ch / 2], [cw / 2, -ch / 2], [cw / 2, ch / 2], [-cw / 2, ch / 2]], np.float32)
    # inclinación (efecto trapecio) + rotación en el plano
    t = tilt
    quad[0, 0] += cw * t * rng.uniform(0.6, 1.0); quad[1, 0] -= cw * t * rng.uniform(0.6, 1.0)
    quad[0, 1] += ch * t * 0.4; quad[1, 1] += ch * t * 0.4
    a = np.deg2rad(rot)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    quad = quad @ R.T + np.array([cx, cy], np.float32)
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(src, quad)
    warped = cv2.warpPerspective(sheet, M, (Wo, Ho), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (Wo, Ho))
    img = np.where(mask[..., None] > 0, warped, table).astype(np.float32)

    # sombra suave (p. ej. del celular / mano)
    if shadow > 0:
        sh = np.zeros((Ho, Wo), np.float32)
        p0 = (int(Wo * rng.uniform(0.0, 0.4)), int(Ho * rng.uniform(0.5, 1.0)))
        cv2.ellipse(sh, p0, (int(Wo * 0.55), int(Ho * 0.35)), rng.uniform(-30, 30), 0, 360, 1.0, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), 90)
        img *= (1 - shadow * sh)[..., None]
    # gradiente de iluminación
    grad = 1.0 - 0.25 * (xx / Wo) * rng.uniform(0.3, 1.0)
    img *= (grad * dim)[..., None]
    if warm:
        img[..., 2] *= 1 + 0.10 * warm; img[..., 0] *= 1 - 0.12 * warm
    img += npr.normal(0, 4, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    return img


CASES = [
    # nombre, N, seed, parámetros de cámara, descripción
    ("4x4_frontal", 4, 11, dict(tilt=0.03, rot=1.5), "4×4 · toma frontal, luz uniforme"),
    ("4x4_inclinado", 4, 23, dict(tilt=0.16, rot=-9), "4×4 · inclinación lateral + rotación"),
    ("5x5_sombra", 5, 37, dict(tilt=0.08, rot=4, shadow=0.45), "5×5 · sombra fuerte del celular"),
    ("5x5_angulo", 5, 41, dict(tilt=0.20, rot=12, warm=1.0), "5×5 · ángulo pronunciado, luz cálida"),
    ("6x6_frontal", 6, 53, dict(tilt=0.05, rot=-3), "6×6 · toma frontal"),
    ("6x6_tenue", 6, 67, dict(tilt=0.12, rot=7, shadow=0.3, dim=0.72), "6×6 · iluminación tenue + sombra"),
]


def main():
    OUT.mkdir(exist_ok=True)
    index = []
    for name, n, seed, cam, desc in CASES:
        rng = random.Random(seed)
        b, sol = make_puzzle(n, rng)
        sheet = render_sheet(b, rng)
        photo = photograph(sheet, rng, **cam)
        cv2.imwrite(str(OUT / f"{name}.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 88])
        thumb = cv2.resize(photo, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT / f"{name}_thumb.jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
        (OUT / f"{name}.json").write_text(json.dumps({"board": b, "solution": sol}, indent=1))
        index.append({"name": name, "N": n, "desc": desc, "image": f"{name}.jpg", "thumb": f"{name}_thumb.jpg"})
        print("ok", name)
    old = OUT / "index.json"
    if old.exists():  # conservar fotos reales añadidas a mano
        index = [e for e in json.loads(old.read_text(encoding="utf-8")) if e.get("real")] + index
    (OUT / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
