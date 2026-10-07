---
title: Futoshiki Solver
emoji: 🧩
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Futoshiki Solver — Visión + YOLOv11-cls + OR-Tools CP-SAT

Aplicación web end-to-end: foto del celular → tablero formal → solución con CP-SAT → solución proyectada sobre la foto.

## Ejecutar
```bash
pip install -r requirements-full.txt   # con PyTorch: carga el .pt con YOLO(path)
python app.py                          # http://127.0.0.1:8000
```
En Windows basta con doble clic en `run.bat` (crea un entorno virtual la primera vez).

El modelo `weights/best.pt` (yolo11n-cls fine-tuneado; clases `1…6`, `vacio`; imgsz 96) se carga automáticamente al iniciar.
Otro modelo: arrástralo al panel **Modelo propio (.pt)**, escribe su ruta, o define `FUTOSHIKI_WEIGHTS=ruta/al/modelo.pt`.

## Desplegar en Vercel
Vercel limita cada función de Python a 500 MB, y PyTorch + Ultralytics no caben. Por eso en Vercel el
backend usa **`weights/best.onnx`** (tu mismo `best.pt` exportado; predicciones idénticas) con
`onnxruntime`, y `requirements.txt` no incluye PyTorch. Todo lo demás (OpenCV, OR-Tools CP-SAT) es igual.

**Opción A — desde GitHub (recomendada)**
1. Sube esta carpeta a un repositorio de GitHub (`app.py`, `vercel.json` y `requirements.txt` en la raíz).
2. En https://vercel.com → **Add New… → Project** → importa el repositorio.
3. Framework Preset: **FastAPI** (Vercel lo detecta solo). No cambies nada más → **Deploy**.

**Opción B — desde la terminal**
```bash
npm i -g vercel
vercel login
vercel --prod        # dentro de esta carpeta
```

Notas:
- Si entrenas un modelo nuevo, vuelve a exportarlo: `python tools/export_onnx.py ruta/al/best.pt`
  (genera `weights/best.onnx` + `weights/best.meta.json`) y despliega de nuevo.
- En Vercel el panel «Modelo propio» acepta `.onnx` (no `.pt`); el archivo subido vive solo en esa
  instancia temporal. Para cambiar el modelo de forma permanente, reemplaza `weights/best.onnx`.
- Las fotos se reducen a 1600 px en el navegador antes de subirse (límite de 4.5 MB por petición).

## Desplegar en Hugging Face Spaces (gratis)
1. Crea una cuenta en https://huggingface.co y entra a **New Space**.
2. Elige un nombre, **SDK: Docker** (plantilla *Blank*) y hardware **CPU basic (free)**.
3. En la pestaña **Files → Add file → Upload files**, arrastra TODO el contenido de esta carpeta
   (no la carpeta en sí: `app.py`, `Dockerfile`, `README.md` deben quedar en la raíz) y confirma.
4. El Space se construye solo (~5–10 min la primera vez). Queda en `https://<usuario>-<nombre>.hf.space`.

Alternativa con Git:
```bash
git clone https://huggingface.co/spaces/<usuario>/<nombre>
# copia los archivos del proyecto dentro, luego:
git add . && git commit -m "Futoshiki Solver" && git push
```

## Desplegar con Docker en cualquier servidor
```bash
docker build -t futoshiki .
docker run -p 7860:7860 futoshiki      # http://localhost:7860
```

## Estructura
```
app.py                 API FastAPI + servidor de la página
pipeline/vision.py     Fase 1 (bilateral, umbral adaptativo, homografía 2 fases, segmentación, ρ, signos) + Fase 3 (M⁻¹)
pipeline/model.py      Carga dinámica: from ultralytics import YOLO; model = YOLO(path)
pipeline/solver.py     Fase 2: CP-SAT, AllDifferent, desigualdades, unicidad, diagnóstico reificado
static/                Frontend (index.html, styles.css, app.js, KaTeX local)
examples/              Casos de prueba con ground truth (.json)
tools/make_examples.py Regenera la galería de casos de prueba
tools/evaluate.py      Evalúa el pipeline: python tools/evaluate.py [yolo|hybrid|ocr]
```

## API
| Método | Ruta | Uso |
|---|---|---|
| GET  | `/api/status` | estado del modelo, metadatos del checkpoint y versiones |
| POST | `/api/model/upload` | sube un `.pt` (multipart `file`) |
| POST | `/api/model/path` | `{"path": "weights/best.pt"}` |
| POST | `/api/solve-image` | multipart `image` o `example`, `engine` (yolo/hybrid/ocr), `n` opcional |
| POST | `/api/solve-board` | `{"board": {...}, "det_id": "..."}` — resuelve (y re-proyecta) un tablero editado |

Formato del tablero: `grid` N×N (0 = vacío), `h` N×(N-1) con `<`/`>`, `v` (N-1)×N con `^` (arriba < abajo) / `v` (arriba > abajo).

## Notas
- La foto debe mostrar las celdas como cajas separadas (con espaciadores entre ellas), como en el dataset.
- El clasificador YOLO lee dígitos y vacíos; los signos se leen con la regla geométrica de apertura de brazos.
- Si la lectura deja el modelo INFEASIBLE, el diagnóstico reificado señala las pistas/signos sospechosos y el editor los marca en rojo.
