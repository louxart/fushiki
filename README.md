Solucionador Automático de Futoshiki
Sistema que resuelve acertijos Futoshiki a partir de una foto tomada con el celular. Lee el tablero con visión computacional, reconoce los dígitos con un modelo YOLOv11 entrenado por el equipo, resuelve el acertijo con Google OR-Tools (CP-SAT) y dibuja la solución sobre la foto original.
Autores
Nicole Yessenia Vasquez Tinco
Alessandro Daniel Bravo Castillo
Alessandro Elías Hesse Pulache
Enlaces
Recurso	URL
Demo desplegada (Vercel)	`<!-- completar: https://xxxx.vercel.app -->`
Repositorio desplegable	`<!-- completar: https://github.com/USUARIO/REPO -->`
Repositorio de notebooks	https://github.com/Crostz/TB1_Topicos_CC
---
Tabla de contenidos
Arquitectura general
Stack tecnológico
Fase 1: Visión y preprocesamiento de la foto
Modelo de reconocimiento de dígitos (YOLOv11-cls)
Lectura de símbolos y armado del JSON
Fase 2: Modelo de Constraint Programming (CP-SAT)
Fase 3: Dibujo de la solución sobre la foto
Despliegue
Ejecución local
Referencias
---
1. Arquitectura general
```
 Foto (celular / PC)
        │
        ▼
┌───────────────────────────┐
│ Fase 1 · Visión (OpenCV)  │  binarización → detección de casillas → homografía → recortes
└───────────────────────────┘
        │
        ├──► Recortes de casillas ──► YOLOv11-cls (best.pt) ──► dígitos impresos
        └──► Recortes entre casillas ──► regla geométrica ──► signos < > ^ v
        │
        ▼
   JSON del estado inicial  ──►  Validador (repetidos / contradicciones)
        │
        ▼
┌───────────────────────────┐
│ Fase 2 · CP-SAT (OR-Tools)│  resolver + verificar unicidad
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ Fase 3 · Dibujo           │  homografía inversa → números en azul sobre la foto
└───────────────────────────┘
        │
        ▼
   Imagen resuelta + JSON + informe de tiempos
```
Todo el flujo (foto → JSON → solver → imagen) es automático, sin intervención manual.
2. Stack tecnológico
Componente	Tecnología
Preprocesamiento de imagen	Python, OpenCV, NumPy
Clasificación de dígitos	Ultralytics YOLOv11 (`yolo11n-cls`, fine-tuning propio)
Solver	Google OR-Tools, CP-SAT
Prototipo / notebooks	Jupyter + Gradio
Interfaz web	`<!-- completar: framework del frontend -->`
Hosting	Vercel `<!-- completar si el backend está en otro servicio -->`
---
3. Fase 1: Visión y preprocesamiento de la foto
Limpieza. La foto se pasa a escala de grises y se binariza con umbral adaptativo, resaltando los contornos aunque haya sombras o luz desigual. Si la foto está borrosa, el paso se repite con parámetros más tolerantes.
Enderezamiento. Los tableros usados no tienen borde exterior: cada casilla es un cuadrado independiente. Por eso se detectan todas las casillas, se toman las esquinas exteriores de las cuatro casillas de las puntas y se aplica una corrección de perspectiva (homografía) para ver el tablero de frente, como si estuviera escaneado. Contando las casillas se obtiene el tamaño del tablero N×N.
Recorte. El tablero enderezado se divide en un recorte por casilla (donde va un número) y en recortes pequeños entre casillas (donde pueden estar los signos de desigualdad).
Descarte de vacíos. Un recorte tiene contenido solo si hay una mancha bastante más oscura que el papel cerca de su centro. Así se ignora la textura del papel y los bordes de las casillas vecinas.
4. Modelo de reconocimiento de dígitos (YOLOv11-cls)
Los dígitos impresos se reconocen con un clasificador de imágenes (no un detector): cada recorte de casilla con contenido entra al modelo y este devuelve el dígito.
Aspecto	Detalle
Arquitectura base	`yolo11n-cls` (Ultralytics YOLOv11, variante nano de clasificación)
Técnica	Fine-tuning (transfer learning) sobre pesos preentrenados
Dataset	Propio: 68 fotos de tableros → 4,839 recortes de casillas
Tamaño de entrada	64 × 64 px
Pesos finales	`best.pt`
Clases	`<!-- completar: p. ej. dígitos 1–6 -->`
Métricas	`<!-- completar: accuracy top-1 en validación -->`
Post-procesamiento: solo se aceptan dígitos válidos para el tamaño del tablero (por ejemplo, nunca un 6 en un tablero de 4×4).
Por qué clasificación y no detección: la Fase 1 ya localiza cada casilla, así que el modelo solo necesita decidir qué dígito hay en un recorte pequeño. Un clasificador nano es más liviano y rápido, lo que facilita el despliegue.
5. Lectura de símbolos y armado del JSON
Signos de desigualdad (OpenCV, regla geométrica). Un signo es angosto en su punta y ancho en su lado abierto, y la punta siempre apunta al número menor. Con esa regla se distinguen los cuatro signos:
`<` y `>` entre casillas de una misma fila.
`^` y `v` entre casillas de una misma columna. Se guardan también como `<` o `>` entre la casilla de arriba y la de abajo.
Estado inicial en JSON. Con dígitos y signos se arma el estado del acertijo:
```json
{
  "grid_size": 5,
  "initial_digits": [{"row": 0, "col": 3, "value": 5}],
  "inequalities": [{"row1": 0, "col1": 1, "row2": 0, "col2": 2, "type": ">"}]
}
```
Validador automático. Antes de resolver, se revisa que la lectura sea coherente. Si un dígito aparece dos veces en una fila o columna, o si un signo contradice dos números leídos (por ejemplo, `2 > 5`), el sistema avisa que algo se leyó mal en la foto.
---
6. Fase 2: Modelo de Constraint Programming (CP-SAT)
La resolución se formula como un problema de satisfacción de restricciones (CSP) y se resuelve con el solver CP-SAT de Google OR-Tools. El modelo es parametrizado: `N`, las pistas y las desigualdades se leen del JSON, por lo que el mismo código resuelve cualquier instancia.
6.1 Datos de entrada
N: tamaño del tablero (`grid_size`).
D: conjunto de pistas `(r, c, v)`: el dígito `v` está impreso en la celda `(r, c)`.
I: conjunto de desigualdades `((r₁, c₁), (r₂, c₂), ⋈)` con `⋈ ∈ {<, >}`, leído como `X(r₁,c₁) ⋈ X(r₂,c₂)`.
6.2 Variables, dominios y restricciones
Elemento	Definición
Variables	`X(i, j)` para `i, j ∈ {0, …, N−1}`: valor de la celda `(i, j)`. Son N² variables enteras.
Dominio	`X(i, j) ∈ {1, …, N}`
Filas (global)	`AllDifferent(X(i,0), …, X(i,N−1))` para cada fila `i`. N restricciones.
Columnas (global)	`AllDifferent(X(0,j), …, X(N−1,j))` para cada columna `j`. N restricciones.
Pistas (unarias)	`X(r, c) = v` para cada `(r, c, v) ∈ D`
Desigualdades (binarias)	`X(r₁,c₁) < X(r₂,c₂)` o `X(r₁,c₁) > X(r₂,c₂)` para cada elemento de `I`
Las restricciones de filas y columnas hacen que la solución sea un cuadrado latino. Se usa `AllDifferent` en lugar de N(N−1)/2 restricciones binarias `X ≠ Y` porque su propagador razona sobre el conjunto completo. Por ejemplo, si dos celdas de una fila solo pueden valer {1, 2}, deduce que ninguna otra celda de esa fila puede ser 1 ni 2. Esa poda no se obtiene con desigualdades binarias por separado.
6.3 Restricciones reificadas (verificación de unicidad)
El modelo del acertijo no necesita reificación, porque todas sus reglas son incondicionales. Una reificación (`b ⇔ C`) solo aporta cuando la validez de una restricción depende de una decisión del modelo, y en Futoshiki eso no ocurre.
Sí se usan para verificar que la solución es única. Tras obtener una solución `s`, se agregan booleanos `b(i, j)` con `b(i, j) ⇔ X(i, j) ≠ s(i, j)` y la cláusula `b(0,0) ∨ … ∨ b(N−1,N−1)` (al menos una celda distinta de `s`). Luego se resuelve de nuevo:
Resultado	Interpretación
Segunda resolución `INFEASIBLE`	La solución es única: la lectura de la foto fue completa.
Existe otra solución	Probablemente faltó leer algún signo o dígito.
Primera resolución `INFEASIBLE`	La lectura contiene un error (dígito o signo mal leído).
6.4 Implementación
```python
from ortools.sat.python import cp_model

m = cp_model.CpModel()
X = [[m.NewIntVar(1, N, f'X_{i}_{j}') for j in range(N)] for i in range(N)]

for i in range(N):
    m.AddAllDifferent(X[i])                          # fila i
    m.AddAllDifferent([X[r][i] for r in range(N)])   # columna i

for d in estado['initial_digits']:
    m.Add(X[d['row']][d['col']] == d['value'])

for q in estado['inequalities']:
    a, b = X[q['row1']][q['col1']], X[q['row2']][q['col2']]
    m.Add(a < b) if q['type'] == '<' else m.Add(a > b)

solver = cp_model.CpSolver()
status = solver.Solve(m)        # OPTIMAL / FEASIBLE / INFEASIBLE
```
6.5 Complejidad y tiempos de respuesta
N² variables con dominio de tamaño N, 2N restricciones `AllDifferent` de aridad N, |D| restricciones unarias y |I| binarias, con |I| ≤ 2N(N−1) (pares de celdas adyacentes).
Espacio de búsqueda sin poda: N^(N²). Para N = 6 son 6³⁶ ≈ 10²⁸ asignaciones.
En el caso general el problema es NP-completo: un Futoshiki sin signos es completar un cuadrado latino parcial (Colbourn, 1984).
Para los tamaños del proyecto (N ≤ 6), la propagación de `AllDifferent` y de las desigualdades, junto con el aprendizaje de cláusulas de CP-SAT, reduce el espacio casi por completo: el solver responde en milisegundos.
---
7. Fase 3: Dibujo de la solución sobre la foto
Con la solución, se aplica la homografía inversa (la misma transformación de la Fase 1, al revés) para calcular la posición y el ángulo exactos de cada casilla en la foto original. Los números faltantes se dibujan en azul sobre los espacios en blanco, respetando la inclinación y la perspectiva del papel.
