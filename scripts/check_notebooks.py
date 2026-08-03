"""Static checks for notebooks, so broken cells never reach a Colab run.

Two bug classes actually escaped to the user before this existed:
  1. a `\\n` escape that became a REAL newline inside a Python string literal,
     making a whole cell unparseable;
  2. a name (`ESTIMATORS`) left referenced in a later cell after being removed
     from the config cell -> NameError halfway through a run.

Both are caught here. Notebooks are runners executed top-to-bottom, so we can
accumulate the names each cell defines and flag loads that were never defined.

Usage:  python scripts/check_notebooks.py [notebooks/*.ipynb]
Exit code 1 if any problem is found.
"""

from __future__ import annotations

import ast
import builtins
import glob
import json
import sys


def _collect_bindings(tree):
    """Names this cell binds (assignments, imports, defs, comprehensions, ...)."""
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Global):
            bound.update(node.names)
    return bound


def check_notebook(path: str) -> list[str]:
    problems: list[str] = []
    nb = json.load(open(path, encoding="utf-8"))
    known = set(dir(builtins)) | {"__name__", "__file__", "get_ipython", "display"}

    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        # skip cells that are entirely commented out / magics-only
        if all(l.strip().startswith(("#", "!", "%")) or not l.strip()
               for l in src.splitlines()):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            problems.append(f"cell {i}: SYNTAX ERROR {e.msg} (line {e.lineno})")
            continue

        loaded = {n.id for n in ast.walk(tree)
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        bound = _collect_bindings(tree)
        undefined = sorted(loaded - bound - known)
        if undefined:
            problems.append(f"cell {i}: possibly undefined at runtime: {undefined}")
        known |= bound
    return problems


def main(argv):
    paths = argv[1:] or sorted(glob.glob("notebooks/*.ipynb"))
    total = 0
    for p in paths:
        probs = check_notebook(p)
        status = "OK" if not probs else f"{len(probs)} PROBLEM(S)"
        print(f"{p}: {status}")
        for msg in probs:
            print(f"    {msg}")
        total += len(probs)
    print(f"\n{total} problem(s) across {len(paths)} notebook(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
