# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import ast
import shutil
import sys

path = Path("nestsar_sms_s1c_v2.py").resolve()
if not path.is_file():
    raise FileNotFoundError(path)

text = path.read_text(encoding="utf-8")
tree = ast.parse(text, filename=str(path))

parent = {}
for node in ast.walk(tree):
    for child in ast.iter_child_nodes(node):
        parent[child] = node

sms_class = None
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == "SpatialMemorySweep":
        sms_class = node
        break

if sms_class is None:
    raise RuntimeError("No encontré class SpatialMemorySweep.")

target_raise = None
for node in ast.walk(sms_class):
    if isinstance(node, ast.Raise):
        segment = ast.get_source_segment(text, node) or ""
        if "Se esperaban 5 partes" in segment:
            target_raise = node
            break

if target_raise is None:
    print("No encontré el raise 'Se esperaban 5 partes'.")
    print("La clase puede estar ya parcheada; mostrando auditoría.")
else:
    p = parent.get(target_raise)
    while p is not None and not isinstance(p, ast.If):
        p = parent.get(p)

    if p is None:
        raise RuntimeError("Encontré el raise P=5, pero no su if contenedor.")
    if not (hasattr(p, "lineno") and hasattr(p, "end_lineno")):
        raise RuntimeError("El AST no expone rango de líneas para el if P=5.")

    lines = text.splitlines(keepends=True)
    start = p.lineno - 1
    end = p.end_lineno
    original_block = "".join(lines[start:end])
    print("=== BLOQUE P=5 ENCONTRADO ===")
    print(original_block.rstrip())
    print("================================")

    backup = path.with_suffix(path.suffix + ".bak_exact_p5_guard")
    if not backup.exists():
        shutil.copy2(path, backup)
        print("Backup creado:", backup)
    else:
        print("Backup ya existente:", backup)

    first_line = lines[start]
    indent = first_line[: len(first_line) - len(first_line.lstrip())]
    replacement = (
        f"{indent}# Dynamic-SMS: acepta cualquier cantidad positiva de tokens espaciales.\n"
        f"{indent}if parts < 1:\n"
        f"{indent}    raise ValueError("
        f'f"Se esperaba al menos 1 token espacial; recibido P={{parts}}"'
        f")\n"
    )

    patched_lines = lines[:start] + [replacement] + lines[end:]
    patched = "".join(patched_lines)
    ast.parse(patched, filename=str(path))
    compile(patched, str(path), "exec")
    path.write_text(patched, encoding="utf-8")
    text = patched
    print("✅ Guard P=5 sustituido por P>=1.")

tree = ast.parse(text, filename=str(path))
sms_class = next(
    (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SpatialMemorySweep"),
    None,
)
if sms_class is None:
    raise RuntimeError("SpatialMemorySweep desapareció tras el parche.")

class_source = ast.get_source_segment(text, sms_class) or ""
errors = []
if "Se esperaban 5 partes" in class_source:
    errors.append("todavía aparece el mensaje 'Se esperaban 5 partes'")

for node in ast.walk(sms_class):
    if isinstance(node, ast.Name) and node.id == "BODY_PARTS":
        errors.append(f"BODY_PARTS todavía usado dentro de SMS en línea {node.lineno}")
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == 5
    ):
        errors.append(f"range(5) todavía usado dentro de SMS en línea {node.lineno}")

print("\n=== AUDITORÍA DE LITERALES '5' EN SpatialMemorySweep ===")
sms_lines = class_source.splitlines()
base = sms_class.lineno
found = False
for i, line in enumerate(sms_lines):
    if "5" in line:
        print(f"L{base + i}: {line}")
        found = True
if not found:
    print("Ninguna línea contiene '5'.")

if errors:
    print("\n❌ QUEDAN RESTRICCIONES SOSPECHOSAS:")
    for err in errors:
        print(" -", err)
    print("\nNo ejecutes todavía el profiler.")
    sys.exit(3)

print("\n✅ SpatialMemorySweep ya no está estructuralmente atado a 5 tokens.")
print("✅ Sintaxis/compile OK.")
