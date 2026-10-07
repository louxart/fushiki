"""
Fase 1 - Visión Computacional & IA  +  Fase 3 - Visualización Aumentada.

1. Pre-proceso: filtro bilateral + binarización adaptativa de Gauss.
2. Corrección proyectiva en 2 fases:
      Fase A (hoja/mesa)     : contorno de la hoja -> homografía M1
      Fase B (celdas)        : cajas de las celdas -> homografía M2 a la vista canónica
      M = M2 · M1 ;  se conserva M⁻¹ para proyectar la solución sobre la foto.
3. Segmentación en celdas y espaciadores (gutters); filtro de densidad de tinta ρ < 0.02 = vacío.
4. Lectura de pistas: YOLOv11-cls propio (.pt) | OCR estructural por correlación normalizada | híbrido.
5. Signos (<, >, ^, v): regla geométrica de apertura de brazos.
"""
from __future__ import annotations

import base64
import time

import cv2
import numpy as np

RHO_EMPTY = 0.02          # densidad de tinta bajo la cual un recorte se considera vacío
CELL = 100                # lado de celda en la vista canónica (px)
MARGIN = 40               # margen de la vista canónica
MAX_SIDE = 1600           # lado máximo de la foto procesada


# =============================================================================== utilidades
def b64img(img, quality=85, max_w=None):
    if img is None:
        return None
    if max_w and img.shape[1] > max_w:
        s = max_w / img.shape[1]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def b64png(img):
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def order_quad(pts):
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(1); d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def binarize(gray, block=None, C=10):
    """Filtro bilateral (preserva bordes) + umbral adaptativo gaussiano (tinta = 255)."""
    f = cv2.bilateralFilter(gray, 9, 50, 50)
    if block is None:
        block = max(15, (min(gray.shape) // 40) | 1)
    return cv2.adaptiveThreshold(f, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, C)


# =============================================================================== Fase A: hoja
def find_sheet(gray):
    """Busca la hoja de papel (cuadrilátero claro más grande). Devuelve 4 esquinas o None."""
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 0.15 * h * w or area > 0.85 * h * w:
        return None          # la hoja ocupa todo el encuadre: no hace falta la fase A
    peri = cv2.arcLength(c, True)
    for eps in (0.02, 0.03, 0.04, 0.05):
        ap = cv2.approxPolyDP(c, eps * peri, True)
        if len(ap) == 4 and cv2.isContourConvex(ap):
            q = order_quad(ap)
            # esquinas pegadas al borde de la foto => la "hoja" es en realidad el encuadre
            m = 0.02 * max(h, w)
            on_edge = sum(1 for x, y in q if x < m or y < m or x > w - m or y > h - m)
            if on_edge >= 2:
                return None
            return q
    return None


def sheet_homography(quad, max_side=1400):
    tl, tr, br, bl = quad
    wA = np.linalg.norm(br - bl); wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br); hB = np.linalg.norm(tl - bl)
    W, H = max(wA, wB), max(hA, hB)
    s = min(1.0, max_side / max(W, H)) if max(W, H) > max_side else 1.0
    W, H = int(W * s), int(H * s)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    return cv2.getPerspectiveTransform(quad, dst), (W, H)


# =============================================================================== Fase B: celdas
def find_boxes(binimg, min_frac=0.025, max_frac=0.3):
    """Contornos cuadriláteros ~cuadrados (las cajas de las celdas)."""
    h, w = binimg.shape
    mside = min(h, w)
    cnts, _ = cv2.findContours(binimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < min_frac * mside or bw > max_frac * mside:
            continue
        ar = bw / float(bh)
        if not 0.7 < ar < 1.4:
            continue
        peri = cv2.arcLength(c, True)
        ap = None
        for eps in (0.04, 0.06, 0.08):
            t = cv2.approxPolyDP(c, eps * peri, True)
            if len(t) == 4 and cv2.isContourConvex(t):
                ap = t; break
        if ap is None:
            continue
        a = cv2.contourArea(ap)
        if a < 0.6 * bw * bh:
            continue
        cands.append({"quad": order_quad(ap), "side": (bw + bh) / 2, "cx": x + bw / 2, "cy": y + bh / 2, "area": a})
    # quitar duplicados (borde exterior/interior de la misma caja) -> conservar el mayor
    cands.sort(key=lambda d: -d["area"])
    kept = []
    for d in cands:
        if all(np.hypot(d["cx"] - k["cx"], d["cy"] - k["cy"]) > 0.35 * k["side"] for k in kept):
            kept.append(d)
    if not kept:
        return []
    # cluster por tamaño: el tamaño con más vecinos (±22 %)
    sides = np.array([k["side"] for k in kept])
    votes = [(np.abs(sides - s) < 0.3 * s).sum() for s in sides]
    ref = sides[int(np.argmax(votes))]
    return [k for k in kept if abs(k["side"] - ref) < 0.3 * ref]


def _cluster_index(vals, pitch):
    """Índices enteros de fila/columna a partir de coordenadas 1-D (huecos = filas faltantes)."""
    order = np.argsort(vals)
    idx = np.zeros(len(vals), int)
    cur, last = 0, vals[order[0]]
    for i in order:
        gap = vals[i] - last
        if gap > 0.5 * pitch:
            cur += max(1, int(round(gap / pitch)))
            last = vals[i]
        idx[i] = cur
    return idx


def index_boxes(boxes):
    """Asigna (fila, columna) a cada caja: orientación dominante + agrupamiento con huecos."""
    C = np.array([[b["cx"], b["cy"]] for b in boxes], np.float64)
    side = float(np.median([b["side"] for b in boxes]))
    angs, nn = [], []
    for i, c in enumerate(C):
        dd = np.hypot(*(C - c).T); dd[i] = 1e9
        nn.append(dd.min())
        for j in np.argsort(dd)[:4]:
            if dd[j] < 2.2 * side:
                v = C[j] - c
                angs.append(np.arctan2(v[1], v[0]))
    pitch = float(np.median(nn))
    a4 = np.array(angs) * 4
    theta = np.arctan2(np.sin(a4).mean(), np.cos(a4).mean()) / 4 if len(angs) else 0.0
    R = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
    P = C @ R.T
    rows = _cluster_index(P[:, 1], pitch)
    cols = _cluster_index(P[:, 0], pitch)
    for b, r, c in zip(boxes, rows, cols):
        b["r"], b["c"] = int(r), int(c)
    return pitch, side


def fit_lattice(boxes, G, margin):
    """Homografía (RANSAC) usando las 4 esquinas de TODAS las cajas indexadas."""
    pitch = CELL + G
    src, dst = [], []
    for b in boxes:
        x0 = margin + b["c"] * pitch; y0 = margin + b["r"] * pitch
        src.extend(b["quad"]); dst.extend([[x0, y0], [x0 + CELL, y0], [x0 + CELL, y0 + CELL], [x0, y0 + CELL]])
    src = np.array(src, np.float32); dst = np.array(dst, np.float32)
    if len(boxes) >= 3:
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
        if H is not None:
            return H
    return cv2.getPerspectiveTransform(src[:4], dst[:4])


def grid_homography(sheet_gray, boxes, n_hint=None):
    """
    Fase B: índices de celda -> homografía con todas las cajas -> re-detección en la vista
    rectificada (con margen de una celda extra) para recuperar cajas/filas que faltaron.
    """
    pitch_px, side = index_boxes(boxes)
    g = float(np.clip(pitch_px / side - 1.0, 0.15, 1.2))
    G = int(round(g * CELL))
    P = CELL + G
    ext = MARGIN + P                       # margen ampliado: cabe una fila/columna extra
    for b in boxes:
        pass
    rmin = min(b["r"] for b in boxes); cmin = min(b["c"] for b in boxes)
    for b in boxes:
        b["r"] -= rmin; b["c"] -= cmin
    H = fit_lattice(boxes, G, ext)
    for _ in range(3):
        n_r = max(b["r"] for b in boxes) + 1; n_c = max(b["c"] for b in boxes) + 1
        n = max(n_r, n_c)
        size = n * P - G + 2 * ext
        warped = cv2.warpPerspective(sheet_gray, H, (size, size), borderValue=255)
        wb = binarize(warped, block=31, C=12)
        cand = find_boxes(wb, min_frac=0.7 * CELL / size, max_frac=1.35 * CELL / size)
        Hinv = np.linalg.inv(H)
        new = []
        for k in cand:
            fr = (k["cy"] - ext - CELL / 2) / P; fc = (k["cx"] - ext - CELL / 2) / P
            r, c = int(round(fr)), int(round(fc))
            if abs(fr - r) > 0.25 or abs(fc - c) > 0.25 or abs(k["side"] - CELL) > 0.3 * CELL:
                continue
            q = cv2.perspectiveTransform(k["quad"].reshape(-1, 1, 2).astype(np.float32), Hinv).reshape(4, 2)
            new.append({"quad": q, "r": r, "c": c, "side": k["side"], "cx": 0, "cy": 0})
        if len(new) < 4:
            break
        rmin = min(b["r"] for b in new); cmin = min(b["c"] for b in new)
        rmax = max(b["r"] for b in new); cmax = max(b["c"] for b in new)
        for b in new:
            b["r"] -= rmin; b["c"] -= cmin
        changed = (rmin, cmin) != (0, 0) or rmax - rmin + 1 != n_r or cmax - cmin + 1 != n_c or len(new) > len(boxes)
        boxes = new
        H = fit_lattice(boxes, G, ext)
        if not changed:
            break
    n = max(max(b["r"] for b in boxes), max(b["c"] for b in boxes)) + 1
    if n_hint:
        n = int(n_hint)
    W = n * CELL + (n - 1) * G
    T = np.array([[1, 0, MARGIN - ext], [0, 1, MARGIN - ext], [0, 0, 1]], np.float64)
    M2 = T @ H
    return M2, G, W + 2 * MARGIN, boxes, n


def cell_rects(n, G, canon_bin):
    """Rectángulos de celda en la vista canónica, refinados con las cajas re-detectadas."""
    pitch = CELL + G
    rects = [[(MARGIN + c * pitch, MARGIN + r * pitch, CELL, CELL) for c in range(n)] for r in range(n)]
    boxes = find_boxes(canon_bin, min_frac=0.6 * CELL / canon_bin.shape[0], max_frac=1.4 * CELL / canon_bin.shape[0])
    for r in range(n):
        for c in range(n):
            x, y, w, h = rects[r][c]
            cx, cy = x + w / 2, y + h / 2
            best = None
            for b in boxes:
                dd = np.hypot(b["cx"] - cx, b["cy"] - cy)
                if dd < 0.3 * CELL and (best is None or dd < best[0]):
                    best = (dd, b)
            if best:
                q = best[1]["quad"]
                x0, y0 = q[:, 0].min(), q[:, 1].min(); x1, y1 = q[:, 0].max(), q[:, 1].max()
                rects[r][c] = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
    return rects


# =============================================================================== lectura
def ink_density(binimg):
    return float((binimg > 0).mean()) if binimg.size else 0.0


def clean_ink(binimg):
    """Elimina motas y componentes pegados al borde del recorte (restos de líneas de caja)."""
    if binimg is None or binimg.ndim != 2 or min(binimg.shape) < 3:
        return np.zeros((1, 1), np.uint8)
    binimg = np.ascontiguousarray(binimg)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(binimg, 8)
    out = np.zeros_like(binimg)
    h, w = binimg.shape
    areas = st[1:, cv2.CC_STAT_AREA] if nl > 1 else np.array([])
    if not len(areas):
        return out
    amax = areas.max()
    for i in range(1, nl):
        x, y, bw, bh, a = st[i]
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if a < max(30, 0.12 * amax):      # motas / textura del papel
            continue
        if touches and (bw > 0.8 * w or bh > 0.8 * h) and a < 0.5 * amax + 1:
            continue
        out[lab == i] = 255
    return out


def make_crop64(gray_crop):
    """Recorte normalizado 64x64 (fondo blanco, tinta oscura) para el clasificador."""
    g = gray_crop.astype(np.float32)
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    # rango mínimo: en recortes vacíos no se amplifica el ruido del papel
    rng = max(hi - lo, 90.0)
    lo = min(lo, hi - rng)
    g = np.clip((g - lo) / rng * 255, 0, 255).astype(np.uint8)
    return cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA)


# ---- OCR estructural por correlación normalizada (respaldo)
_TEMPL = None


def _norm_glyph(binimg, size=32):
    ys, xs = np.nonzero(binimg)
    if len(xs) == 0:
        return None
    crop = binimg[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = max(h, w)
    pad = np.zeros((s, s), np.uint8)
    pad[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    g = cv2.resize(pad, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g, (3, 3), 0).astype(np.float32)


def _templates():
    global _TEMPL
    if _TEMPL is None:
        _TEMPL = {}
        fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX,
                 cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_PLAIN]
        for d in range(1, 10):
            lst = []
            for f in fonts:
                for th in (3, 6, 9):
                    im = np.zeros((140, 140), np.uint8)
                    cv2.putText(im, str(d), (30, 110), f, 3.2 if f != cv2.FONT_HERSHEY_PLAIN else 6, 255, th, cv2.LINE_AA)
                    g = _norm_glyph((im > 100).astype(np.uint8) * 255)
                    if g is not None:
                        lst.append(g)
            _TEMPL[d] = lst
    return _TEMPL


def ocr_digit(binimg, n):
    g = _norm_glyph(binimg)
    if g is None:
        return 0, 0.0, {}
    scores = {}
    for d in range(1, n + 1):
        best = -1
        for t in _templates()[d]:
            v = float(cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)[0, 0])
            best = max(best, v)
        scores[d] = best
    d = max(scores, key=scores.get)
    return d, max(0.0, scores[d]), scores


def plausible_sign(binimg):
    """Descarta manchas: el trazo debe ocupar una fracción razonable del espaciador en ambos ejes."""
    ys, xs = np.nonzero(binimg)
    if len(xs) < 30:
        return False
    h, w = binimg.shape
    bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    return bh >= max(8, 0.18 * h) and bw >= max(8, 0.18 * w)


# ---- signos: regla geométrica de apertura de brazos
def classify_sign(binimg, horizontal):
    """
    El signo es un ángulo: el lado donde los brazos están ABIERTOS tiene mayor dispersión
    perpendicular al eje del espaciador.
      horizontal: dispersión vertical en tercio izquierdo vs derecho  -> abierto a la derecha = '<'
      vertical  : dispersión horizontal en tercio superior vs inferior -> abierto abajo = '^'
    """
    ys, xs = np.nonzero(binimg)
    if len(xs) < 8:
        return "", 0.0, {}
    a, p = (xs, ys) if horizontal else (ys, xs)       # eje principal / perpendicular
    lo, hi = a.min(), a.max()
    L = hi - lo + 1
    first = p[a <= lo + 0.3 * L]; last = p[a >= hi - 0.3 * L]
    sp1 = float(first.max() - first.min() + 1) if len(first) else 0.0
    sp2 = float(last.max() - last.min() + 1) if len(last) else 0.0
    conf = abs(sp2 - sp1) / max(sp1, sp2, 1)
    if horizontal:
        sign = "<" if sp2 > sp1 else ">"
    else:
        sign = "^" if sp2 > sp1 else "v"
    return sign, float(min(1.0, conf * 1.6)), {"apertura_1": sp1, "apertura_2": sp2}


# =============================================================================== pipeline
class Detector:
    def __init__(self, model_manager=None):
        self.mm = model_manager

    def run(self, img_bgr, engine="hybrid", n_hint=None):
        t0 = time.perf_counter()
        timings = {}
        dbg = {}
        h0, w0 = img_bgr.shape[:2]
        sc = min(1.0, MAX_SIDE / max(h0, w0))
        if sc < 1:
            img_bgr = cv2.resize(img_bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        dbg["original"] = img_bgr.copy()
        dbg["binaria"] = cv2.cvtColor(binarize(gray), cv2.COLOR_GRAY2BGR)

        # ---- Fase A: hoja / mesa
        quad = find_sheet(gray)
        if quad is not None:
            M1, (Ws, Hs) = sheet_homography(quad)
            sheet = cv2.warpPerspective(gray, M1, (Ws, Hs))
            vis = img_bgr.copy()
            cv2.polylines(vis, [quad.astype(np.int32)], True, (255, 120, 40), 4, cv2.LINE_AA)
            for p in quad:
                cv2.circle(vis, tuple(int(v) for v in p), 12, (40, 200, 255), -1)
            dbg["fase1_hoja"] = vis
        else:
            M1 = np.eye(3, dtype=np.float64)
            sheet = gray
            dbg["fase1_hoja"] = img_bgr.copy()
        timings["fase1_ms"] = (time.perf_counter() - t0) * 1000

        # ---- Fase B: cajas de celdas
        t1 = time.perf_counter()
        sbin = binarize(sheet)
        boxes = find_boxes(sbin)
        if len(boxes) < 9:
            raise ValueError(f"No se detectó la cuadrícula (cajas encontradas: {len(boxes)}). "
                             "Intenta una foto más cercana, frontal y con buena luz.")
        vis = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
        n_first = len(boxes)
        M2, G, CW, boxes, n = grid_homography(sheet, boxes, n_hint)
        n = int(np.clip(n, 3, 9))
        for b in boxes:
            q = b["quad"].astype(np.int32)
            cv2.polylines(vis, [q], True, (60, 200, 60), 3, cv2.LINE_AA)
            cx, cy = q.mean(0).astype(int)
            cv2.putText(vis, f"{b['r']},{b['c']}", (cx - 18, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 60, 255), 2)
        dbg["fase2_celdas"] = vis
        M = M2 @ M1
        Minv = np.linalg.inv(M)
        canon = cv2.warpPerspective(img_bgr, M, (CW, CW), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
        cgray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)
        cbin = binarize(cgray, block=31, C=15)
        cbin = cv2.morphologyEx(cbin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        rects = cell_rects(n, G, cbin)
        timings["fase2_ms"] = (time.perf_counter() - t1) * 1000

        # ---- Segmentación
        t2 = time.perf_counter()
        seg = canon.copy()
        cells, crops, cell_bins = [], [], []
        for r in range(n):
            row = []
            for c in range(n):
                x, y, w, h = rects[r][c]
                m = int(0.15 * w)
                ib = cbin[y + m:y + h - m, x + m:x + w - m]
                ib = clean_ink(ib)
                gcrop = cgray[y + int(0.08 * h):y + h - int(0.08 * h), x + int(0.08 * w):x + w - int(0.08 * w)]
                rho = ink_density(ib)
                row.append({"r": r, "c": c, "rho": round(rho, 4), "rect": [x, y, w, h]})
                crops.append(make_crop64(gcrop)); cell_bins.append(ib)
                cv2.rectangle(seg, (x, y), (x + w, y + h), (80, 200, 80) if rho >= RHO_EMPTY else (200, 200, 200), 2)
            cells.append(row)

        def gutter(r, c, horiz):
            x, y, w, h = rects[r][c]
            if horiz:
                x2 = rects[r][c + 1][0]
                gx0, gx1 = x + w + 4, x2 - 4
                gy0, gy1 = y + int(0.12 * h), y + h - int(0.12 * h)
            else:
                y2 = rects[r + 1][c][1]
                gy0, gy1 = y + h + 4, y2 - 4
                gx0, gx1 = x + int(0.12 * w), x + w - int(0.12 * w)
            gx0, gy0 = max(gx0, 0), max(gy0, 0)
            return (gx0, gy0, gx1, gy1), clean_ink(cbin[gy0:gy1, gx0:gx1])

        board = {"N": n, "grid": [[0] * n for _ in range(n)],
                 "h": [[""] * (n - 1) for _ in range(n)], "v": [[""] * n for _ in range(n - 1)]}
        sign_conf = {"h": [[None] * (n - 1) for _ in range(n)], "v": [[None] * n for _ in range(n - 1)]}
        gutters_info = []
        for r in range(n):
            for c in range(n - 1):
                (x0, y0, x1, y1), gb = gutter(r, c, True)
                rho = ink_density(gb)
                if rho >= RHO_EMPTY and plausible_sign(gb):
                    s, cf, ex = classify_sign(gb, True)
                    board["h"][r][c] = s; sign_conf["h"][r][c] = round(cf, 3)
                    cv2.rectangle(seg, (x0, y0), (x1, y1), (0, 140, 255), 2)
                    cv2.putText(seg, s, (x0 + 2, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)
                else:
                    cv2.rectangle(seg, (x0, y0), (x1, y1), (225, 225, 225), 1)
                gutters_info.append({"type": "h", "r": r, "c": c, "rho": round(rho, 4), "sign": board["h"][r][c]})
        for r in range(n - 1):
            for c in range(n):
                (x0, y0, x1, y1), gb = gutter(r, c, False)
                rho = ink_density(gb)
                if rho >= RHO_EMPTY and plausible_sign(gb):
                    s, cf, ex = classify_sign(gb, False)
                    board["v"][r][c] = s; sign_conf["v"][r][c] = round(cf, 3)
                    cv2.rectangle(seg, (x0, y0), (x1, y1), (0, 140, 255), 2)
                    cv2.putText(seg, s, (x0 + 2, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)
                else:
                    cv2.rectangle(seg, (x0, y0), (x1, y1), (225, 225, 225), 1)
                gutters_info.append({"type": "v", "r": r, "c": c, "rho": round(rho, 4), "sign": board["v"][r][c]})
        timings["segmentacion_ms"] = (time.perf_counter() - t2) * 1000

        # ---- Lectura de dígitos
        t3 = time.perf_counter()
        used_engine = engine
        yolo = None
        if engine in ("yolo", "hybrid"):
            if self.mm is not None and self.mm.ready:
                yolo = self.mm.classify(crops)
            else:
                used_engine = "ocr"
        cell_conf = [[None] * n for _ in range(n)]
        sources = [[None] * n for _ in range(n)]
        for i, cell in enumerate([c for row in cells for c in row]):
            r, c = cell["r"], cell["c"]
            ink = cell["rho"] >= RHO_EMPTY
            val, conf, src_ = 0, 1.0, "densidad"
            y_lab = y_conf = None
            if yolo is not None:
                y_lab, y_conf, probs = yolo[i]
                # restringir a 1..N (+ vacío)
                if y_lab != "vacio" and int(y_lab) > n:
                    alt = {k: v for k, v in probs.items() if k == "vacio" or int(k) <= n}
                    y_lab = max(alt, key=alt.get); y_conf = alt[y_lab]
                cell["yolo"] = {"label": y_lab, "conf": round(float(y_conf), 4)}
            if used_engine == "yolo":
                if y_lab != "vacio" and ink:
                    val, conf, src_ = int(y_lab), y_conf, "yolo"
                elif ink:
                    val, conf, src_ = 0, y_conf, "yolo"
            elif used_engine == "ocr":
                if ink:
                    d, cf, _ = ocr_digit(cell_bins[i], n)
                    val, conf, src_ = d, cf, "ocr"
            else:  # híbrido
                if ink:
                    if y_lab != "vacio" and y_conf >= 0.60:
                        val, conf, src_ = int(y_lab), y_conf, "yolo"
                    else:
                        d, cf, _ = ocr_digit(cell_bins[i], n)
                        if y_lab != "vacio" and y_conf >= cf:
                            val, conf, src_ = int(y_lab), y_conf, "yolo"
                        elif cf >= 0.45:
                            val, conf, src_ = d, cf, "ocr"
                        else:
                            val, conf, src_ = 0, 1 - cf, "vacío (baja confianza)"
            board["grid"][r][c] = int(val)
            cell_conf[r][c] = round(float(conf), 4)
            sources[r][c] = src_
            if val:
                x, y, w, h = rects[r][c]
                cv2.putText(seg, f"{val}", (x + 4, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 40, 160), 2)
        timings["lectura_ms"] = (time.perf_counter() - t3) * 1000
        dbg["segmentacion"] = seg

        confs = [cell_conf[r][c] for r in range(n) for c in range(n) if board["grid"][r][c]]
        confs += [v for row in sign_conf["h"] for v in row if v is not None]
        confs += [v for row in sign_conf["v"] for v in row if v is not None]
        accuracy = float(np.mean(confs)) if confs else 1.0

        mosaic = self._mosaic(crops, n)
        timings["total_vision_ms"] = (time.perf_counter() - t0) * 1000
        return {
            "board": board,
            "engine_requested": engine,
            "engine_used": used_engine,
            "cell_conf": cell_conf,
            "cell_source": sources,
            "sign_conf": sign_conf,
            "cells": cells,
            "gutters": gutters_info,
            "detection_score": round(accuracy, 4),
            "boxes_found": len(boxes),
            "boxes_first_pass": n_first,
            "gutter_ratio": round(G / CELL, 3),
            "M": M.tolist(),
            "Minv": Minv.tolist(),
            "canon_size": CW,
            "rects": rects,
            "timings": {k: round(v, 1) for k, v in timings.items()},
            "_img": img_bgr, "_canon": canon, "_dbg": dbg, "_mosaic": mosaic, "_Minv": Minv,
        }

    @staticmethod
    def _mosaic(crops, n):
        rows = [np.hstack([cv2.copyMakeBorder(c, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=200) for c in crops[r * n:(r + 1) * n]])
                for r in range(n)]
        return cv2.cvtColor(np.vstack(rows), cv2.COLOR_GRAY2BGR)


# =============================================================================== Fase 3
def _draw_digits(canvas, rects, board, solution, color, alpha_canvas=None):
    n = board["N"]
    for r in range(n):
        for c in range(n):
            if board["grid"][r][c] or not solution:
                continue
            x, y, w, h = rects[r][c]
            txt = str(solution[r][c])
            scale = w / 42
            th = max(2, int(w / 14))
            (tw, tth), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, scale, th)
            org = (int(x + (w - tw) / 2), int(y + (h + tth) / 2))
            cv2.putText(canvas, txt, org, cv2.FONT_HERSHEY_DUPLEX, scale, color, th, cv2.LINE_AA)
            if alpha_canvas is not None:
                cv2.putText(alpha_canvas, txt, org, cv2.FONT_HERSHEY_DUPLEX, scale, 255, th, cv2.LINE_AA)


def render_projection(det, solution):
    """Proyecta la solución sobre la foto original usando M⁻¹ (realidad aumentada)."""
    img = det["_img"]
    H, W = img.shape[:2]
    CW = det["canon_size"]
    board = det["board"]
    color = np.zeros((CW, CW, 3), np.uint8)
    alpha = np.zeros((CW, CW), np.uint8)
    # tinte suave en las celdas resueltas
    for r in range(board["N"]):
        for c in range(board["N"]):
            if not board["grid"][r][c] and solution:
                x, y, w, h = det["rects"][r][c]
                cv2.rectangle(color, (x + 6, y + 6), (x + w - 6, y + h - 6), (120, 230, 140), -1)
                cv2.rectangle(alpha, (x + 6, y + 6), (x + w - 6, y + h - 6), 70, -1)
    _draw_digits(color, det["rects"], board, solution, (60, 150, 20), alpha)
    Minv = det["_Minv"]
    cw = cv2.warpPerspective(color, Minv, (W, H), flags=cv2.INTER_LINEAR)
    aw = cv2.warpPerspective(alpha, Minv, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255
    out = (img.astype(np.float32) * (1 - aw) + cw.astype(np.float32) * aw).astype(np.uint8)
    # contorno de la cuadrícula proyectado
    m, Wg = MARGIN, CW - 2 * MARGIN
    corners = np.array([[[m, m]], [[m + Wg, m]], [[m + Wg, m + Wg]], [[m, m + Wg]]], np.float32)
    pc = cv2.perspectiveTransform(corners, Minv).astype(np.int32)
    cv2.polylines(out, [pc], True, (230, 110, 60), 3, cv2.LINE_AA)
    return out


def render_topview(det, solution):
    canon = det["_canon"].copy()
    _draw_digits(canon, det["rects"], det["board"], solution, (60, 150, 20))
    return canon
