/* Futoshiki Solver — frontend */
"use strict";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

const state = {
  engine: "hybrid",
  res: null,          // respuesta de /api/solve-image
  board: null,        // tablero detectado (posiblemente re-resuelto)
  solve: null,        // último resultado del solver
  exports: null,
  images: null,
  lastInput: null,    // {file} | {example}
  expTab: "json",
  ed: { board: null, detId: null, live: true, solve: null, exports: null, expTab: "json" },
  model: null,
};

// ------------------------------------------------------------------ utilidades
function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg; t.className = "toast show" + (err ? " err" : "");
  clearTimeout(t._h); t._h = setTimeout(() => (t.className = "toast"), 3200);
}
async function api(url, opts = {}) {
  const r = await fetch(url, opts);
  let data = null;
  try { data = await r.json(); } catch { /* */ }
  if (!r.ok) throw new Error((data && (data.detail || data.message)) || `Error ${r.status}`);
  return data;
}
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const fmtInt = (n) => (n == null ? "—" : Number(n).toLocaleString("es-PE"));
function download(name, content, type = "text/plain") {
  const a = document.createElement("a");
  a.href = content.startsWith && content.startsWith("data:") ? content : URL.createObjectURL(new Blob([content], { type }));
  a.download = name; document.body.appendChild(a); a.click(); a.remove();
}
function copy(text) {
  navigator.clipboard.writeText(text).then(() => toast("Copiado al portapapeles"), () => toast("No se pudo copiar", true));
}
function openModal(src, cap = "") {
  $("#modalImg").src = src; $("#modalCap").textContent = cap; $("#modal").classList.add("open");
}
$("#modal").addEventListener("click", () => $("#modal").classList.remove("open"));
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#modal").classList.remove("open"); });

function emptyBoard(n) {
  return { N: n, grid: Array.from({ length: n }, () => Array(n).fill(0)),
    h: Array.from({ length: n }, () => Array(n - 1).fill("")),
    v: Array.from({ length: n - 1 }, () => Array(n).fill("")) };
}
const clone = (o) => JSON.parse(JSON.stringify(o));

// ------------------------------------------------------------------ tema
(function theme() {
  let saved = null;
  try { saved = localStorage.getItem("fz-theme"); } catch { /* */ }
  if (saved) document.documentElement.dataset.theme = saved;
  $("#themeBtn").onclick = () => {
    const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem("fz-theme", t); } catch { /* */ }
  };
})();

// ------------------------------------------------------------------ pestañas
function showTab(name) {
  $$("#nav button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + name));
  window.scrollTo({ top: 0, behavior: "smooth" });
}
$$("#nav button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));

// ------------------------------------------------------------------ render de tablero
const SIGN_PTS = { "<": "16,4 7,12 16,20", ">": "8,4 17,12 8,20", "^": "4,16 12,7 20,16", v: "4,8 12,17 20,8" };
function signSvg(s) {
  if (!s) return "";
  return `<svg viewBox="0 0 24 24"><polyline points="${SIGN_PTS[s]}" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
}
function sizes(n, scale = 1) {
  const cs = Math.round(({ 3: 70, 4: 62, 5: 52, 6: 44, 7: 38 }[n] || 34) * scale);
  return { cs, gs: Math.round(cs * 0.42) };
}
/**
 * opts: values (solución), conf (cell_conf), signConf, editable, suspects:Set, onChange, scale
 */
function renderBoard(el, board, opts = {}) {
  const n = board.N;
  const { cs, gs } = sizes(n, opts.scale || 1);
  const fz = document.createElement("div");
  fz.className = "fz" + (opts.editable ? " edit" : "");
  fz.style.setProperty("--cs", cs + "px"); fz.style.setProperty("--gs", gs + "px");
  const cols = []; for (let i = 0; i < 2 * n - 1; i++) cols.push(i % 2 ? "var(--gs)" : "var(--cs)");
  fz.style.gridTemplateColumns = cols.join(" ");
  fz.style.gridTemplateRows = cols.join(" ");
  const sus = opts.suspects || new Set();
  for (let i = 0; i < 2 * n - 1; i++) {
    for (let j = 0; j < 2 * n - 1; j++) {
      const d = document.createElement("div");
      if (i % 2 === 0 && j % 2 === 0) {
        const r = i / 2, c = j / 2;
        const given = board.grid[r][c];
        const val = given || (opts.values ? opts.values[r][c] : 0);
        d.className = "cell" + (!given && val ? " calc" : "");
        if (sus.has(`g:${r},${c}`)) d.classList.add("bad");
        const cf = opts.conf && opts.conf[r] ? opts.conf[r][c] : null;
        if (given && cf != null && cf < 0.7) { d.classList.add("low"); d.title = `Confianza ${(cf * 100).toFixed(0)} % — revisar`; }
        else if (given && cf != null) d.title = `Confianza ${(cf * 100).toFixed(1)} %`;
        if (opts.editable) {
          const inp = document.createElement("input");
          inp.inputMode = "numeric"; inp.maxLength = 1; inp.value = given || "";
          inp.setAttribute("aria-label", `Celda fila ${r + 1} columna ${c + 1}`);
          inp.oninput = () => {
            const v = parseInt(inp.value, 10);
            board.grid[r][c] = v >= 1 && v <= n ? v : 0;
            inp.value = board.grid[r][c] || "";
            opts.onChange && opts.onChange();
          };
          inp.onkeydown = (e) => navCells(e, fz, r, c, n);
          inp.dataset.rc = `${r},${c}`;
          d.appendChild(inp);
        } else {
          d.textContent = val || "";
        }
      } else if (i % 2 === 0) {
        const r = i / 2, c = (j - 1) / 2;
        const s = board.h[r][c];
        d.className = "gut" + (s ? "" : " empty");
        d.innerHTML = signSvg(s);
        const sc = opts.signConf && opts.signConf.h[r] ? opts.signConf.h[r][c] : null;
        if (s && sc != null && sc < 0.35) d.classList.add("low");
        if (sus.has(`h:${r},${c}`)) d.classList.add("bad");
        if (opts.editable) d.onclick = () => { board.h[r][c] = { "": "<", "<": ">", ">": "" }[s]; opts.onChange && opts.onChange(true); };
      } else if (j % 2 === 0) {
        const r = (i - 1) / 2, c = j / 2;
        const s = board.v[r][c];
        d.className = "gut" + (s ? "" : " empty");
        d.innerHTML = signSvg(s);
        const sc = opts.signConf && opts.signConf.v[r] ? opts.signConf.v[r][c] : null;
        if (s && sc != null && sc < 0.35) d.classList.add("low");
        if (sus.has(`v:${r},${c}`)) d.classList.add("bad");
        if (opts.editable) d.onclick = () => { board.v[r][c] = { "": "^", "^": "v", v: "" }[s]; opts.onChange && opts.onChange(true); };
      }
      fz.appendChild(d);
    }
  }
  el.innerHTML = ""; el.appendChild(fz);
}
function navCells(e, fz, r, c, n) {
  const mv = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }[e.key];
  if (!mv) return;
  const rr = Math.min(n - 1, Math.max(0, r + mv[0])), cc = Math.min(n - 1, Math.max(0, c + mv[1]));
  const t = fz.querySelector(`input[data-rc="${rr},${cc}"]`);
  if (t) { e.preventDefault(); t.focus(); t.select(); }
}
function emptyState(el, msg) {
  el.innerHTML = `<div class="empty-state"><svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg><div>${msg}</div></div>`;
}

// SVG vectorial limpio
function boardSvg(board, sol) {
  const n = board.N, C = 60, G = 26, M = 20, W = n * C + (n - 1) * G + 2 * M;
  let s = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${W}" width="${W}" height="${W}" font-family="Inter,Arial,sans-serif">`;
  s += `<rect width="${W}" height="${W}" rx="14" fill="#ffffff"/>`;
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const x = M + c * (C + G), y = M + r * (C + G);
    const g = board.grid[r][c], v = g || (sol ? sol[r][c] : 0);
    s += `<rect x="${x}" y="${y}" width="${C}" height="${C}" rx="5" fill="${g ? "#eef1ff" : "#fff"}" stroke="#1c2140" stroke-width="2"/>`;
    if (v) s += `<text x="${x + C / 2}" y="${y + C / 2 + 11}" text-anchor="middle" font-size="31" font-weight="700" fill="${g ? "#1c2140" : "#3346c8"}">${v}</text>`;
  }
  const chev = (cx, cy, sign) => {
    const k = 8, pts = { "<": [[k, -k], [-k, 0], [k, k]], ">": [[-k, -k], [k, 0], [-k, k]], "^": [[-k, k * .7], [0, -k * .7], [k, k * .7]], v: [[-k, -k * .7], [0, k * .7], [k, -k * .7]] }[sign];
    return `<polyline points="${pts.map((p) => `${cx + p[0]},${cy + p[1]}`).join(" ")}" fill="none" stroke="#1c2140" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>`;
  };
  for (let r = 0; r < n; r++) for (let c = 0; c < n - 1; c++) if (board.h[r][c]) s += chev(M + c * (C + G) + C + G / 2, M + r * (C + G) + C / 2, board.h[r][c]);
  for (let r = 0; r < n - 1; r++) for (let c = 0; c < n; c++) if (board.v[r][c]) s += chev(M + c * (C + G) + C / 2, M + r * (C + G) + C + G / 2, board.v[r][c]);
  return s + "</svg>";
}

// ------------------------------------------------------------------ JSON resaltado
function hlJson(obj) {
  const txt = typeof obj === "string" ? obj : JSON.stringify(obj, compactArrays, 2);
  return esc(txt)
    .replace(/(&quot;|")([^"\n]*?)("|&quot;)(\s*:)/g, '<span class="k">"$2"</span>$4')
    .replace(/: "([^"]*)"/g, ': <span class="s">"$1"</span>')
    .replace(/\b(-?\d+\.?\d*)\b/g, '<span class="n">$1</span>');
}
function compactArrays(k, v) { return v; }
function prettyBoardJson(o) {
  // filas de matrices en una sola línea
  let s = JSON.stringify(o, null, 2);
  s = s.replace(/\[\s+([^\[\]{}]*?)\s+\]/g, (m, inner) => "[" + inner.replace(/\s*\n\s*/g, " ") + "]");
  return s;
}

// ------------------------------------------------------------------ stepper
function step(i, cls, sub) {
  const st = $("#st" + i);
  st.className = "st " + (cls || "");
  st.querySelector("small").textContent = sub;
  const ic = st.querySelector(".ic");
  if (cls === "done") ic.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5" stroke="#fff" stroke-width="2.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  else if (cls === "run") ic.innerHTML = '<svg class="spin" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6"><path d="M21 12a9 9 0 1 1-6.2-8.6" stroke-linecap="round"/></svg>';
  else if (cls === "err") ic.textContent = "!";
  else ic.textContent = i;
  if (i < 3) $("#ln" + i).classList.toggle("done", cls === "done");
}
function resetSteps() {
  step(1, "", "Esperando archivo"); step(2, "", "Tablero sin reconocer"); step(3, "", "Pendiente");
  $("#readyTime").textContent = "—";
}

// ------------------------------------------------------------------ imagen
const dropImg = $("#dropImg");
$("#pickImg").onclick = (e) => { e.stopPropagation(); $("#imgInput").click(); };
dropImg.onclick = () => $("#imgInput").click();
$("#imgInput").onchange = (e) => { const f = e.target.files[0]; if (f) runImage({ file: f }); e.target.value = ""; };
["dragenter", "dragover"].forEach((ev) => dropImg.addEventListener(ev, (e) => { e.preventDefault(); dropImg.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dropImg.addEventListener(ev, (e) => { e.preventDefault(); dropImg.classList.remove("drag"); }));
dropImg.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (!f) return;
  if (/\.(pt|onnx)$/i.test(f.name)) return uploadModel(f);
  runImage({ file: f });
});
$("#clearImg").onclick = () => {
  $("#loadedBox").hidden = true; state.res = null; state.lastInput = null; resetSteps();
  emptyState($("#detBoard"), "Sube una foto o elige un caso de prueba para detectar el tablero.");
  emptyState($("#solBoard"), "La solución aparecerá aquí.");
};
$("#zoomImg").onclick = () => $("#loadedImg").src && openModal($("#loadedImg").src, $("#loadedName").textContent);
$("#origBtn").onclick = () => {
  if (state.images) openModal(state.images.original, "Imagen original");
  else if ($("#loadedImg").src) openModal($("#loadedImg").src);
  else toast("Aún no hay imagen cargada");
};

$$("#engineSeg button").forEach((b) => (b.onclick = () => {
  $$("#engineSeg button").forEach((x) => x.classList.toggle("active", x === b));
  state.engine = b.dataset.engine;
  if (state.engine === "yolo" && !(state.model && state.model.ready)) toast("No hay modelo .pt cargado: se usará OCR como respaldo", true);
}));

// Reduce la foto en el navegador (lado mayor 1600 px, JPEG 0.9): el backend trabaja a esa
// resolución igualmente y así se respeta el límite de 4.5 MB por petición de Vercel.
async function shrinkImage(file, maxSide = 1600) {
  try {
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    const k = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
    if (k === 1 && file.size < 3.5 * 1024 * 1024) return file;
    const cv = document.createElement("canvas");
    cv.width = Math.round(bmp.width * k); cv.height = Math.round(bmp.height * k);
    cv.getContext("2d").drawImage(bmp, 0, 0, cv.width, cv.height);
    const blob = await new Promise((res) => cv.toBlob(res, "image/jpeg", 0.9));
    return blob ? new File([blob], file.name.replace(/\.[^.]+$/, "") + ".jpg", { type: "image/jpeg" }) : file;
  } catch { return file; }
}

async function runImage(input) {
  if (input.file && input.file.size > 25 * 1024 * 1024) return toast("La imagen supera 25 MB", true);
  state.lastInput = input;
  const t0 = performance.now();
  // vista previa
  $("#loadedBox").hidden = false;
  if (input.file) {
    $("#loadedImg").src = URL.createObjectURL(input.file);
    $("#loadedName").textContent = input.file.name;
    $("#loadedSize").textContent = `${Math.round(input.file.size / 1024)} KB`;
  } else {
    $("#loadedImg").src = "/examples/" + input.example;
    $("#loadedName").textContent = input.example;
    $("#loadedSize").textContent = "caso de prueba";
  }
  step(1, "done", "Archivo procesado"); step(2, "run", "Rectificando y leyendo…"); step(3, "", "Pendiente");
  $("#detLoad").classList.add("on");
  const fd = new FormData();
  if (input.file) fd.append("image", await shrinkImage(input.file)); else fd.append("example", input.example);
  fd.append("engine", state.engine);
  const n = $("#nHint").value; if (n) fd.append("n", n);
  try {
    const res = await api("/api/solve-image", { method: "POST", body: fd });
    state.res = res; state.board = res.detection.board; state.solve = res.solve;
    state.exports = res.exports; state.images = res.images;
    const eng = { yolo: "YOLO propio", hybrid: "Híbrido", ocr: "OCR" }[res.detection.engine_used];
    step(2, "done", `Tablero reconocido · ${eng}`);
    if (res.solve.solved) step(3, "done", res.solve.unique ? "Solución única encontrada" : "Solución encontrada (no única)");
    else step(3, "err", "Inconsistente · ver diagnóstico");
    $("#readyTime").textContent = `${((performance.now() - t0) / 1000).toFixed(2)} segundos`;
    renderAll();
    if (res.detection.engine_requested !== res.detection.engine_used) toast("Sin modelo .pt cargado: se usó OCR estructural");
  } catch (err) {
    step(2, "err", "No se detectó la cuadrícula"); step(3, "", "Pendiente");
    emptyState($("#detBoard"), esc(err.message));
    toast(err.message, true);
  } finally {
    $("#detLoad").classList.remove("on");
  }
}
$("#retryBtn").onclick = () => state.lastInput ? runImage(state.lastInput) : toast("Primero sube una imagen");

// ------------------------------------------------------------------ render principal
function renderAll() {
  const res = state.res, d = res ? res.detection : null, b = state.board, s = state.solve;
  if (!b) return;
  const n = b.N;
  $("#sizeBadge").textContent = `${n} × ${n}`;
  renderBoard($("#detBoard"), b, { conf: d && d.cell_conf, signConf: d && d.sign_conf });
  $("#detLegend").hidden = false;
  if (d) {
    const nY = d.cell_source.flat().filter((x) => x === "yolo").length;
    const nO = d.cell_source.flat().filter((x) => x === "ocr").length;
    const mname = state.model && state.model.path ? state.model.path.split(/[\\/]/).pop() : "—";
    $("#detEngine").innerHTML = d.engine_used === "ocr"
      ? `Motor: <b>OCR estructural</b> (sin modelo .pt)`
      : `Motor: <b>${d.engine_used === "yolo" ? "YOLO propio" : "Híbrido"}</b> · modelo <code>${esc(mname)}</code> · ${nY} pista(s) leída(s) por YOLO${nO ? `, ${nO} por OCR` : ""} · ${d.boxes_found} cajas`;
  }
  const sol = s.solution || (s.diagnosis && s.diagnosis.repaired_solution);
  $("#solWrap").classList.toggle("solved", !!s.solved);
  if (sol) {
    const sus = suspectsSet(s);
    renderBoard($("#solBoard"), b, { values: sol, suspects: sus });
    $("#solLegend").hidden = false;
  } else emptyState($("#solBoard"), "Sin solución.");
  const sb = $("#solvedBadge");
  if (s.solved) { sb.className = "badge green"; sb.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24"><circle cx="12" cy="12" r="11" fill="var(--green)"/><path d="M7 12.5l3.2 3.2L17 9" stroke="#fff" stroke-width="2.4" fill="none" stroke-linecap="round"/></svg>¡Resuelto!'; }
  else { sb.className = "badge red"; sb.textContent = "Inconsistente"; }

  // stats
  $("#sSize").textContent = `${n} × ${n}`; $("#sSizeSub").textContent = `${n * n} celdas · ${2 * n * (n - 1)} espaciadores`;
  if (d) { $("#sAcc").textContent = `${(d.detection_score * 100).toFixed(1)} %`; $("#sAccSub").textContent = `${d.boxes_found} cajas · motor ${d.engine_used.toUpperCase()}`; }
  const t = s.telemetry;
  $("#sTime").textContent = `${t.time_ms.toFixed(1)} ms`;
  $("#sTimeSub").textContent = res ? `visión ${res.detection.timings.total_vision_ms} ms · total ${res.timing.total_ms} ms` : "CP-SAT";
  $("#sUniq").textContent = s.solved ? (s.unique ? "Sí" : "No") : "—";
  $("#sUniqSub").textContent = s.solved ? (s.unique ? "Una sola solución posible" : "Existen soluciones alternativas") : "modelo inconsistente";

  renderVisuals();
  renderTelemetry($("#teleKv"), $("#teleStatus"), $("#teleSub"), $("#teleDiag"), s);
  renderExport();
  renderDebug();
}
function suspectsSet(s) {
  const set = new Set();
  if (s && s.diagnosis && s.diagnosis.suspects) for (const k of s.diagnosis.suspects)
    set.add((k.kind === "given" ? "g" : k.kind) + `:${k.r},${k.c}`);
  return set;
}
function renderVisuals() {
  const im = state.images, b = state.board, s = state.solve;
  const sol = s.solution || (s.diagnosis && s.diagnosis.repaired_solution);
  if (im && im.projection) {
    $("#visProj").innerHTML = `<img src="${im.projection}" alt="Foto con solución proyectada">`;
    $("#visProj img").onclick = () => openModal(im.projection, "Solución proyectada con M⁻¹");
  }
  if (im && im.topview) {
    $("#visTop").innerHTML = `<img src="${im.topview}" alt="Vista cenital">`;
    $("#visTop img").onclick = () => openModal(im.topview, "Vista canónica cenital");
  }
  $("#visSvg").innerHTML = boardSvg(b, sol);
  $("#visSvg svg").removeAttribute("width"); $("#visSvg svg").removeAttribute("height");
  $("#visSvg svg").style.width = "100%"; $("#visSvg svg").style.maxWidth = "340px"; $("#visSvg svg").style.cursor = "default";
}
function renderTelemetry(kvEl, statusEl, subEl, diagEl, s) {
  const t = s.telemetry;
  statusEl.textContent = t.status;
  statusEl.className = "badge " + (t.status === "OPTIMAL" || t.status === "FEASIBLE" ? "green" : "red");
  const uniq = s.solved ? (s.unique ? "ÚNICA" : "MÚLTIPLE") : "—";
  kvEl.innerHTML = [
    ["Estado", t.status], ["Unicidad", uniq], ["Tiempo", `${t.time_ms.toFixed(1)} ms`],
    ["Ramas", fmtInt(t.branches)], ["Conflictos", fmtInt(t.conflicts)], ["Variables / restricciones", `${t.num_vars} / ${t.num_constraints}`],
  ].map(([k, v]) => `<div><small>${k}</small><b>${v}</b></div>`).join("");
  let sub = `${t.givens} pistas · ${t.inequalities} desigualdades · 2N AllDifferent.`;
  if (t.uniqueness_status) sub += ` Corte de unicidad Σ[X≠Sol₁] ≥ 1 → <b>${t.uniqueness_status}</b> en ${t.uniqueness_time_ms} ms (${fmtInt(t.uniqueness_branches)} ramas, ${fmtInt(t.uniqueness_conflicts)} conflictos)` + (s.unique ? " ⇒ solución única demostrada." : " ⇒ existe otra solución.");
  subEl.innerHTML = sub;
  diagEl.innerHTML = "";
  if (s.diagnosis) {
    const dg = s.diagnosis;
    diagEl.innerHTML = `<div class="diag"><h5>Diagnóstico por restricciones reificadas · max Σ Bᵢ = ${dg.satisfied ?? "?"} / ${dg.total ?? "?"}</h5>
      El tablero leído no tiene solución. Restricciones que el solver tuvo que relajar (probables errores de cámara):
      <ul>${(dg.suspects || []).map((k) => `<li>${esc(k.id)}</li>`).join("") || "<li>—</li>"}</ul>
      <div style="margin-top:6px">Corrígelas en el <a href="#" data-goto-editor>Editor Manual</a>; la solución mostrada es la reparada.</div></div>`;
    const a = diagEl.querySelector("[data-goto-editor]"); if (a) a.onclick = (e) => { e.preventDefault(); openEditorFromDet(); };
  } else if (s.solved && !s.unique && s.alternative) {
    diagEl.innerHTML = `<div class="diag" style="border-color:var(--amber);background:var(--amber-soft)"><h5 style="color:var(--amber)">Solución no única</h5>Faltan pistas o se perdió algún signo en la lectura; revisa el tablero en el editor.</div>`;
  }
}
function exportText(ex, tab, res) {
  if (!ex) return "—";
  if (tab === "json") return hlJson(prettyBoardJson(ex.json));
  if (tab === "python") return esc(ex.python);
  if (tab === "ascii") return esc("# Tablero leído\n" + ex.ascii + (ex.ascii_solution ? "\n\n# Solución\n" + ex.ascii_solution : ""));
  if (tab === "detail" && res) {
    const d = res.detection;
    const rows = d.cells.flat().map((c) => `(${c.r},${c.c})  ρ=${c.rho.toFixed(3).padEnd(6)} ${String(d.board.grid[c.r][c.c] || "·").padEnd(2)} fuente=${(d.cell_source[c.r][c.c] || "").padEnd(10)} conf=${d.cell_conf[c.r][c.c] != null ? d.cell_conf[c.r][c.c].toFixed(3) : "—"}${c.yolo ? `  yolo=${c.yolo.label}:${c.yolo.conf.toFixed(3)}` : ""}`);
    return esc(`# motor solicitado=${d.engine_requested}  usado=${d.engine_used}\n# cajas=${d.boxes_found}  g (espaciador/celda)=${d.gutter_ratio}\n# tiempos ${JSON.stringify(d.timings)}\n# M = ${JSON.stringify(d.M.map((r) => r.map((x) => +x.toFixed(4))))}\n\n` + rows.join("\n"));
  }
  return "—";
}
function renderExport() { $("#expPre").innerHTML = exportText(state.exports, state.expTab, state.res); }
$$("#expTabs button").forEach((b) => (b.onclick = () => {
  $$("#expTabs button").forEach((x) => x.classList.toggle("active", x === b)); state.expTab = b.dataset.exp; renderExport();
}));
function rawExport(ex, tab) {
  if (!ex) return "";
  if (tab === "json") return prettyBoardJson(ex.json);
  if (tab === "python") return ex.python;
  if (tab === "ascii") return ex.ascii + (ex.ascii_solution ? "\n\n" + ex.ascii_solution : "");
  return $("#expPre").textContent;
}
$("#copyExport").onclick = () => state.exports ? copy(rawExport(state.exports, state.expTab)) : toast("Aún no hay resultado");
$("#dlExport").onclick = () => {
  if (!state.exports) return toast("Aún no hay resultado");
  const ext = { json: "json", python: "py", ascii: "txt", detail: "txt" }[state.expTab];
  download(`futoshiki_${state.expTab}.${ext}`, rawExport(state.exports, state.expTab));
};

// descargar / copiar solución
$("#dlBtn").onclick = (e) => { e.stopPropagation(); $("#dlMenu").classList.toggle("open"); };
document.addEventListener("click", () => $("#dlMenu").classList.remove("open"));
$$("#dlMenu button").forEach((b) => (b.onclick = () => {
  if (!state.exports) return toast("Aún no hay solución");
  const k = b.dataset.dl, ex = state.exports;
  if (k === "json") download("futoshiki_solucion.json", prettyBoardJson(ex.json), "application/json");
  if (k === "py") download("futoshiki_solucion.py", ex.python);
  if (k === "txt") download("futoshiki_solucion.txt", ex.ascii + "\n\n" + (ex.ascii_solution || ""));
  if (k === "png") state.images && state.images.projection ? download("futoshiki_proyeccion.jpg", state.images.projection) : toast("No hay foto");
  if (k === "svg") download("futoshiki_solucion.svg", boardSvg(state.board, ex.json.solution), "image/svg+xml");
}));
$("#copyBtn").onclick = () => state.exports && state.exports.ascii_solution ? copy(state.exports.ascii_solution) : toast("Aún no hay solución");

// resolver de nuevo (CP-SAT sobre el tablero actual)
$("#resolveBtn").onclick = async () => {
  if (!state.board) return toast("Primero detecta un tablero");
  try {
    const out = await api("/api/solve-board", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ board: state.board, det_id: state.res && state.res.det_id }) });
    state.solve = out.solve; state.exports = out.exports;
    if (out.images) Object.assign(state.images, out.images);
    $("#readyTime").textContent = `${(out.timing.total_ms / 1000).toFixed(3)} segundos`;
    step(3, out.solve.solved ? "done" : "err", out.solve.solved ? "Solución encontrada" : "Inconsistente");
    renderAll(); toast(`CP-SAT: ${out.solve.telemetry.status} en ${out.solve.telemetry.time_ms.toFixed(1)} ms`);
  } catch (e) { toast(e.message, true); }
};
$("#editBtn").onclick = () => openEditorFromDet();

// ------------------------------------------------------------------ depuración (pestaña pipeline)
function renderDebug() {
  const im = state.images; if (!im) return;
  $("#dbgEmpty").hidden = true;
  const items = [
    ["original", "1 · Foto original", "entrada de cámara"],
    ["binaria", "2 · Bilateral + umbral adaptativo", "tinta = blanco"],
    ["fase1_hoja", "3 · Fase A: contorno de la hoja", "esquinas → M₁"],
    ["fase2_celdas", "4 · Fase B: cajas de celdas", "esquinas exteriores → M₂"],
    ["segmentacion", "5 · Vista canónica segmentada", "verde = celda con tinta · naranja = signo"],
    ["mosaic", "6 · Recortes 64×64 al clasificador", "entrada de YOLO"],
    ["topview", "7 · Vista cenital resuelta", "render en espacio canónico"],
    ["projection", "8 · Proyección con M⁻¹", "realidad aumentada"],
  ];
  $("#dbgGrid").innerHTML = items.filter(([k]) => im[k]).map(([k, t, s]) =>
    `<figure><img src="${im[k]}" data-cap="${t}" alt="${t}"><figcaption><b>${t}</b><br>${s}</figcaption></figure>`).join("");
  $$("#dbgGrid img").forEach((i) => (i.onclick = () => openModal(i.src, i.dataset.cap)));
}

// ------------------------------------------------------------------ modelo
async function loadStatus() {
  try {
    const st = await api("/api/status");
    state.model = st.model; state.versions = st.versions;
    renderModel();
  } catch (e) {
    $("#modelPillTxt").textContent = "Backend sin conexión"; $("#modelPill").className = "model-pill err";
  }
}
function renderModel() {
  const m = state.model, pill = $("#modelPill"), badge = $("#modelBadge");
  if (m && m.ready) {
    pill.className = "model-pill ok"; $("#modelPillTxt").textContent = "YOLO listo";
    badge.className = "badge green"; badge.textContent = "cargado";
    const names = Object.values(m.names || {});
    $("#modelBox").innerHTML = `
      <div class="row"><span>Archivo</span><span>${esc(m.path.split(/[\\/]/).slice(-2).join("/"))}</span></div>
      <div class="row"><span>Tarea</span><span>${m.task} · imgsz ${m.imgsz ?? "—"}</span></div>
      <div class="row"><span>Backend</span><span>${m.backend === "onnxruntime" ? "ONNX Runtime (sin PyTorch)" : "Ultralytics YOLO (PyTorch)"}</span></div>
      <div class="row"><span>Carga</span><span>${m.load_ms} ms</span></div>
      <div class="classes">${names.map((n) => `<span>${esc(n)}</span>`).join("")}</div>`;
  } else {
    pill.className = "model-pill" + (m && m.error ? " err" : ""); $("#modelPillTxt").textContent = "Sin modelo";
    badge.className = "badge gray"; badge.textContent = "sin cargar";
    $("#modelBox").innerHTML = `<span class="muted">${m && m.error ? esc(m.error) : "Carga tus pesos .pt para activar el motor YOLO. Mientras tanto se usa OCR estructural."}</span>`;
  }
  renderTraining();
}
$("#modelPill").onclick = () => { showTab("solver"); $("#dropPt").scrollIntoView({ behavior: "smooth", block: "center" }); };
async function uploadModel(file) {
  const fd = new FormData(); fd.append("file", file);
  $("#modelBadge").className = "badge amber"; $("#modelBadge").textContent = "cargando…";
  try { state.model = await api("/api/model/upload", { method: "POST", body: fd }); renderModel(); toast(`Modelo cargado: ${Object.keys(state.model.names).length} clases`); }
  catch (e) { toast(e.message, true); loadStatus(); }
}
$("#pickPt").onclick = (e) => { e.stopPropagation(); $("#ptInput").click(); };
$("#dropPt").onclick = () => $("#ptInput").click();
$("#ptInput").onchange = (e) => { const f = e.target.files[0]; if (f) uploadModel(f); e.target.value = ""; };
const dropPt = $("#dropPt");
["dragenter", "dragover"].forEach((ev) => dropPt.addEventListener(ev, (e) => { e.preventDefault(); dropPt.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dropPt.addEventListener(ev, (e) => { e.preventDefault(); dropPt.classList.remove("drag"); }));
dropPt.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) uploadModel(f); });
$("#loadPath").onclick = async () => {
  const p = $("#ptPath").value.trim(); if (!p) return;
  $("#modelBadge").className = "badge amber"; $("#modelBadge").textContent = "cargando…";
  try {
    state.model = await api("/api/model/path", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path: p }) });
    renderModel(); toast("Modelo cargado desde " + p);
  } catch (e) { toast(e.message, true); loadStatus(); }
};

// ------------------------------------------------------------------ galería
async function loadGallery() {
  try {
    const list = await api("/api/examples");
    $("#gallery").innerHTML = list.map((e) => `<button class="gal-item" data-ex="${e.image}"><img src="/examples/${e.thumb}" alt="" loading="lazy"><div><b>${e.N}×${e.N}</b> ${esc(e.desc.split("·")[1] || e.desc)}</div></button>`).join("");
    $$("#gallery .gal-item").forEach((b) => (b.onclick = () => runImage({ example: b.dataset.ex })));
  } catch { $("#gallery").innerHTML = '<span class="muted">Sin ejemplos</span>'; }
}

// ------------------------------------------------------------------ editor manual
function openEditorFromDet() {
  if (!state.board) { showTab("editor"); return toast("No hay detección; empieza desde un tablero vacío"); }
  state.ed.board = clone(state.board); state.ed.detId = state.res && state.res.det_id;
  $("#edN").value = state.ed.board.N;
  showTab("editor"); renderEditor(); edSolve();
}
function renderEditor() {
  const sus = suspectsSet(state.ed.solve);
  renderBoard($("#edBoard"), state.ed.board, { editable: true, scale: 1.25, suspects: sus, onChange: (rerender) => { if (rerender) renderEditor(); edChanged(); } });
}
let edTimer = null;
function edChanged() { if (state.ed.live) { clearTimeout(edTimer); edTimer = setTimeout(edSolve, 250); } }
async function edSolve() {
  const b = state.ed.board;
  try {
    const out = await api("/api/solve-board", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ board: b, det_id: state.ed.detId }) });
    state.ed.solve = out.solve; state.ed.exports = out.exports;
    const s = out.solve;
    const sol = s.solution || (s.diagnosis && s.diagnosis.repaired_solution);
    $("#edSolWrap").classList.toggle("solved", !!s.solved);
    if (sol) renderBoard($("#edSol"), b, { values: sol, suspects: suspectsSet(s) }); else emptyState($("#edSol"), "Sin solución");
    const badge = $("#edBadge");
    if (s.solved) { badge.className = "badge green"; badge.textContent = s.unique ? "¡Resuelto! · única" : "Resuelto · múltiple"; }
    else { badge.className = "badge red"; badge.textContent = "INFEASIBLE"; }
    $("#edResSub").textContent = `CP-SAT ${s.telemetry.status} en ${s.telemetry.time_ms.toFixed(1)} ms`;
    renderTelemetry($("#edKv"), document.createElement("span"), document.createElement("div"), $("#edDiag"), s);
    if (out.images && out.images.projection) {
      $("#edProjCard").hidden = false;
      $("#edProj").innerHTML = `<img src="${out.images.projection}" alt="Re-proyección">`;
      $("#edProj img").onclick = () => openModal(out.images.projection, "Re-proyección tras la edición manual");
    } else $("#edProjCard").hidden = true;
    renderEdExport();
    // resaltar sospechosas sin perder el foco
    const focused = document.activeElement && document.activeElement.dataset ? document.activeElement.dataset.rc : null;
    renderEditor();
    if (focused) { const t = $(`#edBoard input[data-rc="${focused}"]`); if (t) { t.focus(); t.setSelectionRange(1, 1); } }
  } catch (e) { toast(e.message, true); }
}
function renderEdExport() { $("#edExpPre").innerHTML = exportText(state.ed.exports, state.ed.expTab); }
$$("#edExpTabs button").forEach((b) => (b.onclick = () => {
  $$("#edExpTabs button").forEach((x) => x.classList.toggle("active", x === b)); state.ed.expTab = b.dataset.exp; renderEdExport();
}));
$("#edN").onchange = () => { state.ed.board = emptyBoard(+$("#edN").value); state.ed.detId = null; state.ed.solve = null; renderEditor(); edSolve(); };
$("#edClear").onclick = () => { state.ed.board = emptyBoard(state.ed.board.N); state.ed.solve = null; renderEditor(); edSolve(); };
$("#edBad").onclick = () => {
  state.ed.board = { N: 4, grid: [[1, 0, 3, 0], [4, 0, 0, 2], [0, 0, 0, 0], [0, 0, 1, 0]],
    h: [["<", "", ">"], [">", "", ">"], ["<", "", ">"], [">", "", "<"]],
    v: [["v", "", "v", "v"], ["v", "", "v", "v"], ["", "", "", ""]] };
  state.ed.detId = null; $("#edN").value = 4; renderEditor(); edSolve();
};
$("#edFromDet").onclick = () => openEditorFromDet();
$("#edLive").onclick = () => { state.ed.live = !state.ed.live; $("#edLive").classList.toggle("on", state.ed.live); };
$("#edSolve").onclick = () => edSolve();

// ------------------------------------------------------------------ gráficos (dataset)
const tip = $("#tip");
function showTip(e, html) { tip.innerHTML = html; tip.style.display = "block"; tip.style.left = e.clientX + 12 + "px"; tip.style.top = e.clientY - 34 + "px"; }
function hideTip() { tip.style.display = "none"; }

const CLASSES = [
  ["Vacío (ρ < 0.02)", 3265, "celdas y espaciadores sin tinta"],
  ["Dígitos impresos 1–5", 792, "pistas numéricas"],
  ["Signo ‘>’ mayor horizontal", 221, ""],
  ["Signo ‘∧’ menor vertical", 203, ""],
  ["Signo ‘∨’ mayor vertical", 193, ""],
  ["Signo ‘<’ menor horizontal", 165, ""],
];
function barChart(el, data) {
  const total = 4839, W = 560, rowH = 34, L = 190, R = 60, H = data.length * rowH + 28;
  const max = Math.max(...data.map((d) => d[1]));
  const x = (v) => L + (v / max) * (W - L - R);
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Distribución de recortes por clase">`;
  [0, 1000, 2000, 3000].forEach((t) => { s += `<line class="grid" x1="${x(t)}" x2="${x(t)}" y1="0" y2="${H - 22}"/><text x="${x(t)}" y="${H - 6}" text-anchor="middle" font-size="11">${fmtInt(t)}</text>`; });
  data.forEach((d, i) => {
    const y = i * rowH + 6, w = Math.max(2, x(d[1]) - L);
    s += `<text x="${L - 10}" y="${y + 16}" text-anchor="end">${esc(d[0])}</text>`;
    s += `<path class="bar" data-i="${i}" d="M${L} ${y + 4} h${w - 4} a4 4 0 0 1 4 4 v12 a4 4 0 0 1 -4 4 h-${w - 4} z"/>`;
    s += `<text class="lbl" x="${L + w + 8}" y="${y + 16}">${fmtInt(d[1])}</text>`;
    s += `<rect x="0" y="${y}" width="${W}" height="${rowH}" fill="transparent" data-hit="${i}"/>`;
  });
  s += `<line class="axis" x1="${L}" x2="${L}" y1="0" y2="${H - 22}"/></svg>`;
  el.innerHTML = s;
  $$("[data-hit]", el).forEach((r) => {
    const i = +r.dataset.hit, bar = el.querySelector(`.bar[data-i="${i}"]`);
    r.onmousemove = (e) => { bar.classList.add("hl"); showTip(e, `<b>${esc(data[i][0])}</b><br>${fmtInt(data[i][1])} recortes · ${(data[i][1] / total * 100).toFixed(1)} %`); };
    r.onmouseleave = () => { bar.classList.remove("hl"); hideTip(); };
  });
  $("#tblClasses").innerHTML = `<tr><th>Clase</th><th style="text-align:right">Recortes</th><th style="text-align:right">%</th></tr>` +
    data.map((d) => `<tr><td>${esc(d[0])}</td><td class="num">${fmtInt(d[1])}</td><td class="num">${(d[1] / total * 100).toFixed(1)}</td></tr>`).join("") +
    `<tr><td><b>Total</b></td><td class="num"><b>4,839</b></td><td class="num">100.0</td></tr>`;
}
function lineChart(el, xs, series, o = {}) {
  const W = 560, H = 210, L = 46, R = 70, T = 12, B = 30;
  const all = series.flatMap((s) => s.values);
  const yMin = o.yMin ?? Math.min(...all), yMax = o.yMax ?? Math.max(...all);
  const x = (v) => L + (v - xs[0]) / (xs[xs.length - 1] - xs[0] || 1) * (W - L - R);
  const y = (v) => T + (1 - (v - yMin) / (yMax - yMin || 1)) * (H - T - B);
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.title || "")}">`;
  for (let k = 0; k <= 4; k++) { const v = yMin + (yMax - yMin) * k / 4; s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end" font-size="11">${o.fmt ? o.fmt(v) : v.toFixed(2)}</text>`; }
  xs.forEach((v, i) => { if (i % 5 === 0 || i === xs.length - 1) s += `<text x="${x(v)}" y="${H - 10}" text-anchor="middle" font-size="11">${v}</text>`; });
  s += `<text x="${W - R}" y="${H - 10}" text-anchor="start" font-size="11" dx="8">época</text>`;
  series.forEach((se) => {
    s += `<path d="${se.values.map((v, i) => `${i ? "L" : "M"}${x(xs[i]).toFixed(1)} ${y(v).toFixed(1)}`).join(" ")}" fill="none" style="stroke:${se.color}" stroke-width="2" stroke-linejoin="round"/>`;
    const lv = se.values[se.values.length - 1];
    s += `<text x="${x(xs[xs.length - 1]) + 8}" y="${y(lv) + 4 + (se.dy || 0)}" class="lbl" font-size="11.5">${esc(se.name)}</text>`;
  });
  s += `<line class="cross" x1="0" x2="0" y1="${T}" y2="${H - B}" stroke="var(--ink-3)" stroke-dasharray="3 3" visibility="hidden"/>`;
  s += `<g class="dots"></g><rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent" class="hit"/></svg>`;
  el.innerHTML = s;
  const svg = el.querySelector("svg"), cross = svg.querySelector(".cross"), dots = svg.querySelector(".dots");
  svg.querySelector(".hit").onmousemove = (e) => {
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());
    let i = Math.round((p.x - L) / (W - L - R) * (xs.length - 1)); i = Math.max(0, Math.min(xs.length - 1, i));
    cross.setAttribute("x1", x(xs[i])); cross.setAttribute("x2", x(xs[i])); cross.setAttribute("visibility", "visible");
    dots.innerHTML = series.map((se) => `<circle cx="${x(xs[i])}" cy="${y(se.values[i])}" r="4.5" style="fill:${se.color};stroke:var(--card)" stroke-width="2"/>`).join("");
    showTip(e, `<b>Época ${xs[i]}</b><br>` + series.map((se) => `<span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${se.color};margin-right:5px"></span>${esc(se.name)}: ${o.fmt ? o.fmt(se.values[i]) : se.values[i].toFixed(4)}`).join("<br>"));
  };
  svg.querySelector(".hit").onmouseleave = () => { cross.setAttribute("visibility", "hidden"); dots.innerHTML = ""; hideTip(); };
  if (series.length > 1) {
    el.insertAdjacentHTML("afterbegin", `<div class="legend" style="justify-content:flex-start;margin:0 0 4px">${series.map((se) => `<span><i style="background:${se.color};border-color:${se.color}"></i>${esc(se.name)}</span>`).join("")}</div>`);
  }
}
function renderTraining() {
  const tr = state.model && state.model.training;
  const tbl = $("#tblTrain");
  if (!tr) { $("#chartLoss").innerHTML = '<p class="muted">Carga un modelo para ver sus curvas.</p>'; $("#chartAcc").innerHTML = ""; tbl.innerHTML = ""; return; }
  const r = tr.train_results || {};
  if (r.epoch && r.epoch.length) {
    lineChart($("#chartLoss"), r.epoch, [
      { name: "train/loss", color: "var(--s1)", values: r["train/loss"], dy: -9 },
      { name: "val/loss", color: "var(--s2)", values: r["val/loss"], dy: 11 },
    ], { yMin: 0, title: "Pérdida por época", fmt: (v) => v.toFixed(2) });
    lineChart($("#chartAcc"), r.epoch, [{ name: "accuracy top-1", color: "var(--s1)", values: r["metrics/accuracy_top1"] }],
      { title: "Accuracy top-1 por época", fmt: (v) => (v * 100).toFixed(1) + " %" });
    $("#trainSub").innerHTML = `Leídas del checkpoint <code>.pt</code> · ${r.epoch.length} épocas ejecutadas (parada temprana)`;
  }
  const a = tr.train_args || {}, m = tr.train_metrics || {};
  const rows = [
    ["Modelo base", a.model], ["Dataset", a.data], ["Épocas (máx.)", a.epochs], ["imgsz", a.imgsz], ["Batch", a.batch],
    ["Optimizador", a.optimizer], ["lr0", a.lr0], ["Paciencia", a.patience], ["Accuracy top-1 (val)", m["metrics/accuracy_top1"] != null ? (m["metrics/accuracy_top1"] * 100).toFixed(2) + " %" : null],
    ["val/loss", m["val/loss"]], ["Fecha", (tr.date || "").slice(0, 19).replace("T", " ")], ["Ultralytics", tr.ultralytics_version],
    ["Clases", Object.values(state.model.names || {}).join(", ")],
    ["Versiones backend", state.versions ? Object.entries(state.versions).map(([k, v]) => `${k} ${v}`).join(" · ") : null],
  ].filter((x) => x[1] != null && x[1] !== "");
  tbl.innerHTML = rows.map(([k, v]) => `<tr><th>${k}</th><td class="mono" style="font-size:12.5px">${esc(v)}</td></tr>`).join("");
}

// ------------------------------------------------------------------ fórmulas LaTeX
const TEX = {
  "m-thr": String.raw`I_b=\operatorname{bilateral}(I),\qquad T(x,y)=\sum_{(i,j)\in W}G_\sigma(i,j)\,I_b(x{+}i,y{+}j)-C,\qquad B(x,y)=\begin{cases}1 & I_b(x,y)<T(x,y)\\ 0 & \text{en otro caso}\end{cases}`,
  "m-rho": String.raw`\rho(R)=\frac{1}{|R|}\sum_{p\in R}B(p),\qquad \rho(R)<0.02\;\Rightarrow\;\text{vacío}`,
  "m-hom": String.raw`\begin{bmatrix}x'\\y'\\w\end{bmatrix}=M\begin{bmatrix}x\\y\\1\end{bmatrix},\quad (u,v)=\Big(\tfrac{x'}{w},\tfrac{y'}{w}\Big),\qquad M=M_2\,M_1,\qquad p_{\text{foto}}\sim M^{-1}\,p_{\text{canónica}}`,
  "m-arm": String.raw`a_k=\max_{p\in T_k}p_\perp-\min_{p\in T_k}p_\perp,\quad s=\begin{cases}\text{“<” o “}\wedge\text{”} & a_2>a_1\\ \text{“>” o “}\vee\text{”} & a_2<a_1\end{cases},\quad \text{conf}=\frac{|a_2-a_1|}{\max(a_1,a_2)}`,
  "m-ncc": String.raw`\operatorname{NCC}(g,t)=\frac{\sum(g-\bar g)(t-\bar t)}{\sqrt{\sum(g-\bar g)^2\sum(t-\bar t)^2}},\qquad \hat d=\arg\max_{d\le N}\;\max_{t\in\mathcal T_d}\operatorname{NCC}(g,t)`,
  "m-vars": String.raw`\begin{aligned}&\mathcal I=\{0,\dots,N-1\},\qquad X_{r,c}\in\{1,\dots,N\}\quad\forall\,r,c\in\mathcal I\\ &\mathcal G\subseteq\mathcal I^2:\ \text{pistas con valor } g_{r,c}\\ &\mathcal H^{<},\mathcal H^{>}\subseteq\mathcal I\times\{0..N{-}2\}:\ \text{desigualdades horizontales}\\ &\mathcal V^{\wedge},\mathcal V^{\vee}\subseteq\{0..N{-}2\}\times\mathcal I:\ \text{desigualdades verticales}\end{aligned}`,
  "m-alldiff": String.raw`\begin{aligned}&\operatorname{AllDifferent}(X_{r,0},\dots,X_{r,N-1})&&\forall\,r\in\mathcal I\\ &\operatorname{AllDifferent}(X_{0,c},\dots,X_{N-1,c})&&\forall\,c\in\mathcal I\end{aligned}`,
  "m-ineq": String.raw`\begin{aligned}&X_{r,c}=g_{r,c} &&\forall (r,c)\in\mathcal G\\ &X_{r,c}<X_{r,c+1} &&\forall (r,c)\in\mathcal H^{<}\qquad X_{r,c}>X_{r,c+1}\ \ \forall (r,c)\in\mathcal H^{>}\\ &X_{r,c}<X_{r+1,c} &&\forall (r,c)\in\mathcal V^{\wedge}\qquad X_{r,c}>X_{r+1,c}\ \ \forall (r,c)\in\mathcal V^{\vee}\end{aligned}`,
  "m-uniq": String.raw`\begin{aligned}&d_{r,c}\Leftrightarrow\big(X_{r,c}\neq S^{(1)}_{r,c}\big)\\ &\sum_{r,c}d_{r,c}\;\ge\;1\\ &\text{INFEASIBLE}\;\Rightarrow\;S^{(1)}\text{ es la única solución}\end{aligned}`,
  "m-reif": String.raw`\begin{aligned}&B_i\in\{0,1\},\qquad B_i\Leftrightarrow C_i\qquad\forall\,i\in\mathcal C=\mathcal G\cup\mathcal H^{<}\cup\mathcal H^{>}\cup\mathcal V^{\wedge}\cup\mathcal V^{\vee}\\ &\max\ \sum_{i\in\mathcal C}B_i\quad\text{s.a. AllDifferent (filas y columnas) como restricciones duras}\\ &\text{Sospechosas}=\{\,i\in\mathcal C : B_i^{*}=0\,\}\qquad\text{(basta }B_i\Rightarrow C_i\text{: el óptimo fija }B_i=1\text{ si }C_i\text{ se cumple)}\end{aligned}`,
};
function renderTex() {
  for (const [id, tex] of Object.entries(TEX)) {
    const el = document.getElementById(id); if (!el) continue;
    if (window.katex) { try { katex.render(tex, el, { displayMode: true, throwOnError: false }); continue; } catch { /* */ } }
    el.innerHTML = `<pre class="code">${esc(tex)}</pre>`;
  }
}

// ------------------------------------------------------------------ init
resetSteps();
emptyState($("#detBoard"), "Sube una foto o elige un caso de prueba para detectar el tablero.");
emptyState($("#solBoard"), "La solución aparecerá aquí.");
state.ed.board = emptyBoard(4);
renderEditor();
barChart($("#chartClasses"), CLASSES);
renderTex();
loadStatus();
loadGallery();
