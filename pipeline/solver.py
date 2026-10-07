"""
Fase 2 - Razonamiento y Constraint Programming (Google OR-Tools CP-SAT).

Representación del tablero (estado formal):
    N      : tamaño de la cuadrícula
    grid   : N x N, 0 = vacío, 1..N = pista impresa
    h      : N x (N-1), signo entre (r,c) y (r,c+1): '' | '<' | '>'
    v      : (N-1) x N, signo entre (r,c) y (r+1,c): '' | '^' | 'v'
             '^' => X[r][c] < X[r+1][c]   (menor vertical)
             'v' => X[r][c] > X[r+1][c]   (mayor vertical)
"""
from __future__ import annotations

import time
from ortools.sat.python import cp_model


# --------------------------------------------------------------------------- utilidades
def empty_board(n: int) -> dict:
    return {
        "N": n,
        "grid": [[0] * n for _ in range(n)],
        "h": [[""] * (n - 1) for _ in range(n)],
        "v": [[""] * n for _ in range(n - 1)],
    }


def normalize_board(b: dict) -> dict:
    n = int(b["N"])
    grid = [[int(x or 0) for x in row] for row in b["grid"]]
    h = [[(s or "").strip() for s in row] for row in b.get("h", [[""] * (n - 1)] * n)]
    v = [[(s or "").strip() for s in row] for row in b.get("v", [[""] * n] * (n - 1))]
    h = [[s if s in ("<", ">") else "" for s in row] for row in h]
    v = [[("^" if s in ("^", "∧", "A") else "v" if s in ("v", "V", "∨") else "") for s in row] for row in v]
    return {"N": n, "grid": grid, "h": h, "v": v}


def constraint_list(b: dict) -> list[dict]:
    """Lista plana de restricciones 'blandas' (pistas + desigualdades) con id legible."""
    n = b["N"]
    out = []
    for r in range(n):
        for c in range(n):
            if b["grid"][r][c]:
                out.append({"id": f"pista({r},{c})={b['grid'][r][c]}", "kind": "given",
                            "r": r, "c": c, "value": b["grid"][r][c]})
    for r in range(n):
        for c in range(n - 1):
            s = b["h"][r][c]
            if s:
                out.append({"id": f"X[{r},{c}] {s} X[{r},{c+1}]", "kind": "h", "r": r, "c": c, "sign": s})
    for r in range(n - 1):
        for c in range(n):
            s = b["v"][r][c]
            if s:
                op = "<" if s == "^" else ">"
                out.append({"id": f"X[{r},{c}] {op} X[{r+1},{c}]", "kind": "v", "r": r, "c": c, "sign": s})
    return out


def _add_soft(model, X, k, enforce=None):
    """Añade la restricción k; si enforce (BoolVar) se pasa, queda reificada B_i => C_i."""
    def ct(expr):
        c = model.Add(expr)
        if enforce is not None:
            c.OnlyEnforceIf(enforce)
        return c

    if k["kind"] == "given":
        ct(X[k["r"]][k["c"]] == k["value"])
    elif k["kind"] == "h":
        a, b = X[k["r"]][k["c"]], X[k["r"]][k["c"] + 1]
        ct(a < b) if k["sign"] == "<" else ct(a > b)
    else:
        a, b = X[k["r"]][k["c"]], X[k["r"] + 1][k["c"]]
        ct(a < b) if k["sign"] == "^" else ct(a > b)


def _base_model(n):
    m = cp_model.CpModel()
    X = [[m.NewIntVar(1, n, f"X_{r}_{c}") for c in range(n)] for r in range(n)]
    for i in range(n):
        m.AddAllDifferent(X[i])
        m.AddAllDifferent([X[r][i] for r in range(n)])
    return m, X


STATUS = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE",
          cp_model.INFEASIBLE: "INFEASIBLE", cp_model.MODEL_INVALID: "MODEL_INVALID",
          cp_model.UNKNOWN: "UNKNOWN"}


def _run(model, time_limit=10.0):
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = 8
    t0 = time.perf_counter()
    st = s.Solve(model)
    ms = (time.perf_counter() - t0) * 1000
    return s, st, ms


# --------------------------------------------------------------------------- API principal
def solve(board: dict, check_unique: bool = True, diagnose: bool = True) -> dict:
    b = normalize_board(board)
    n = b["N"]
    cons = constraint_list(b)

    model, X = _base_model(n)
    for k in cons:
        _add_soft(model, X, k)

    s, st, ms = _run(model)
    tele = {
        "status": STATUS.get(st, str(st)),
        "time_ms": round(ms, 2),
        "wall_time_ms": round(s.WallTime() * 1000, 2),
        "branches": s.NumBranches(),
        "conflicts": s.NumConflicts(),
        "num_vars": n * n,
        "num_constraints": 2 * n + len(cons),
        "givens": sum(1 for k in cons if k["kind"] == "given"),
        "inequalities": sum(1 for k in cons if k["kind"] != "given"),
    }
    result = {"board": b, "telemetry": tele, "solution": None, "unique": None,
              "diagnosis": None, "solved": False}

    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sol = [[int(s.Value(X[r][c])) for c in range(n)] for r in range(n)]
        result["solution"] = sol
        result["solved"] = True
        if check_unique:
            result.update(_uniqueness(b, cons, sol, tele))
        return result

    # ---- INFEASIBLE: diagnóstico con restricciones reificadas
    if diagnose and st == cp_model.INFEASIBLE:
        result["diagnosis"] = _diagnose(b, cons)
    return result


def _uniqueness(b, cons, sol, tele):
    """Excluye Sol_1 con SUM(X_rc != Sol1_rc) >= 1; si es INFEASIBLE la solución es única."""
    n = b["N"]
    model, X = _base_model(n)
    for k in cons:
        _add_soft(model, X, k)
    diff = []
    for r in range(n):
        for c in range(n):
            d = model.NewBoolVar(f"d_{r}_{c}")
            model.Add(X[r][c] != sol[r][c]).OnlyEnforceIf(d)
            model.Add(X[r][c] == sol[r][c]).OnlyEnforceIf(d.Not())
            diff.append(d)
    model.Add(sum(diff) >= 1)
    s, st, ms = _run(model)
    tele["uniqueness_status"] = STATUS.get(st, str(st))
    tele["uniqueness_time_ms"] = round(ms, 2)
    tele["uniqueness_branches"] = s.NumBranches()
    tele["uniqueness_conflicts"] = s.NumConflicts()
    alt = None
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        alt = [[int(s.Value(X[r][c])) for c in range(n)] for r in range(n)]
    return {"unique": st == cp_model.INFEASIBLE, "alternative": alt}


def _diagnose(b, cons):
    """max SUM B_i  s.a.  B_i => C_i ; AllDifferent duro. Las C_i con B_i=0 son sospechosas."""
    n = b["N"]
    model, X = _base_model(n)
    B = []
    for i, k in enumerate(cons):
        bi = model.NewBoolVar(f"B_{i}")
        _add_soft(model, X, k, enforce=bi)
        B.append(bi)
    model.Maximize(sum(B))
    s, st, ms = _run(model)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": STATUS.get(st, str(st)), "time_ms": round(ms, 2), "suspects": []}
    suspects = [cons[i] for i, bi in enumerate(B) if not s.Value(bi)]
    repaired = [[int(s.Value(X[r][c])) for c in range(n)] for r in range(n)]
    return {
        "status": STATUS.get(st, str(st)),
        "time_ms": round(ms, 2),
        "satisfied": int(s.ObjectiveValue()),
        "total": len(cons),
        "suspects": suspects,
        "repaired_solution": repaired,
    }


# --------------------------------------------------------------------------- exportes
def to_python_matrix(b: dict) -> str:
    return "grid = [\n" + ",\n".join("    " + str(row) for row in b["grid"]) + "\n]"


def to_ascii(b: dict, values=None) -> str:
    n = b["N"]
    vals = values or b["grid"]
    lines = []
    for r in range(n):
        row = ""
        for c in range(n):
            x = vals[r][c]
            row += f"[{x if x else ' '}]"
            if c < n - 1:
                row += f" {b['h'][r][c] or ' '} "
        lines.append(row)
        if r < n - 1:
            vr = ""
            for c in range(n):
                vr += f" {b['v'][r][c] or ' '} "
                if c < n - 1:
                    vr += "   "
            lines.append(vr.rstrip())
    return "\n".join(lines)
