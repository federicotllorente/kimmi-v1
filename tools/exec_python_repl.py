"""Evalúa código Python restringido (cálculos, conversiones, lógica rápida).

Ejecuta en subproceso con timeout. Solo imports de módulos allowlisted.
Requiere: stdlib (usa el mismo intérprete `sys.executable`).
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
import os

TIMEOUT_SEC = 5
MAX_CODE_CHARS = 4000
MAX_OUTPUT_CHARS = 2000

ALLOWED_MODULES = frozenset(
    {
        "math",
        "decimal",
        "datetime",
        "json",
        "re",
        "fractions",
        "statistics",
        "cmath",
        "collections",
        "itertools",
        "functools",
        "typing",
    }
)

_BLOCKED_PATTERNS = [
    re.compile(r"\b__import__\b"),
    re.compile(r"\bopen\s*\("),
    re.compile(r"\bexec\s*\("),
    re.compile(r"\beval\s*\("),
    re.compile(r"\bcompile\s*\("),
    re.compile(r"\bgetattr\s*\("),
    re.compile(r"\bsetattr\s*\("),
    re.compile(r"\bdelattr\s*\("),
    re.compile(r"\bglobals\s*\("),
    re.compile(r"\blocals\s*\("),
    re.compile(r"\bvars\s*\("),
    re.compile(r"\bbreakpoint\s*\("),
    re.compile(r"\binput\s*\("),
    re.compile(r"\bos\s*\."),
    re.compile(r"\bsys\s*\."),
    re.compile(r"\bsubprocess\b"),
    re.compile(r"\bsocket\b"),
    re.compile(r"\bpathlib\b"),
    re.compile(r"\bshutil\b"),
    re.compile(r"\bpickle\b"),
    re.compile(r"\bctypes\b"),
]


def exec_python_repl(code: str) -> str:
    """Ejecuta código Python seguro y devuelve stdout/stderr truncado."""
    if not code or not str(code).strip():
        return "Debes indicar código Python a evaluar."

    source = str(code).strip()
    if len(source) > MAX_CODE_CHARS:
        return f"Código demasiado largo (máximo {MAX_CODE_CHARS} caracteres)."

    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(source):
            return "Código rechazado: contiene operaciones no permitidas."

    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as e:
        return f"Error de sintaxis: {e}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            else:
                root = (node.module or "").split(".")[0]
                if root:
                    names = [root]
            for name in names:
                if name not in ALLOWED_MODULES:
                    return (
                        f"Import no permitido: '{name}'. "
                        f"Permitidos: {', '.join(sorted(ALLOWED_MODULES))}."
                    )
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in ("__builtins__", "__builtin__"):
                return "Código rechazado: acceso a builtins no permitido."

    runner = _build_runner_script(source)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(runner)
            tmp_path = tmp.name
    except Exception as e:
        return f"No se pudo preparar el entorno REPL: {e}"

    try:
        completed = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
            cwd=tempfile.gettempdir(),
            env={
                "PATH": os.environ.get("PATH", ""),
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
                "PYTHONIOENCODING": "utf-8",
            },
        )
    except subprocess.TimeoutExpired:
        return f"Tiempo de ejecución agotado ({TIMEOUT_SEC}s)."
    except Exception as e:
        return f"Error al ejecutar el REPL: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    out = (completed.stdout or "").strip()
    err = (completed.stderr or "").strip()
    if completed.returncode != 0:
        msg = err or out or f"Falló con código {completed.returncode}."
        return _truncate(f"Error: {msg}")

    if not out and not err:
        return "Ejecutado sin salida."
    if err and out:
        return _truncate(f"{out}\n[stderr]: {err}")
    return _truncate(out or err)


def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + "... [truncado]"
    return text


def _build_runner_script(user_code: str) -> str:
    """Envuelve el código del usuario en un namespace mínimo con prints del último expr."""
    # Si la última sentencia es una expresión, imprimir su valor (estilo REPL).
    try:
        tree = ast.parse(user_code, mode="exec")
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = tree.body[-1]
            expr_code = ast.unparse(last.value)
            body_nodes = tree.body[:-1]
            preamble = ast.unparse(ast.Module(body=body_nodes, type_ignores=[])) if body_nodes else ""
            parts = []
            if preamble.strip():
                parts.append(preamble)
            parts.append(f"_result = ({expr_code})")
            parts.append("if _result is not None:\n    print(repr(_result))")
            wrapped_user = "\n".join(parts)
        else:
            wrapped_user = user_code
    except Exception:
        wrapped_user = user_code

    allowed = ", ".join(repr(m) for m in sorted(ALLOWED_MODULES))
    return f'''# Auto-generated restricted runner
import builtins as _b

_ALLOWED = {{{allowed}}}
_orig_import = _b.__import__

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in _ALLOWED:
        raise ImportError(f"Import no permitido: {{name}}")
    return _orig_import(name, globals, locals, fromlist, level)

_safe_builtins = {{
    "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
    "bytearray": bytearray, "bytes": bytes, "chr": chr, "complex": complex,
    "dict": dict, "divmod": divmod, "enumerate": enumerate, "filter": filter,
    "float": float, "format": format, "frozenset": frozenset, "hex": hex,
    "int": int, "isinstance": isinstance, "issubclass": issubclass,
    "iter": iter, "len": len, "list": list, "map": map, "max": max,
    "min": min, "next": next, "oct": oct, "ord": ord, "pow": pow,
    "print": print, "range": range, "repr": repr, "reversed": reversed,
    "round": round, "set": set, "slice": slice, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "type": type, "zip": zip,
    "True": True, "False": False, "None": None,
    "__import__": _safe_import,
}}

_ns = {{"__builtins__": _safe_builtins}}
exec(compile({wrapped_user!r}, "<repl>", "exec"), _ns)
'''
