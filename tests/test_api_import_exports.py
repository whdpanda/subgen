from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API_SRC = ROOT / "src" / "subgen" / "api"
SRC_ROOT = ROOT / "src"


def _module_name(py_file: Path) -> str:
    rel = py_file.relative_to(SRC_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _module_exports(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    explicit_all: set[str] | None = None
    names: set[str] = set()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                    if target.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                        values: list[str] = []
                        if all(isinstance(elt, ast.Constant) and isinstance(elt.value, str) for elt in node.value.elts):
                            values = [elt.value for elt in node.value.elts]  # type: ignore[attr-defined]
                            explicit_all = set(values)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)

    return explicit_all if explicit_all is not None else names


def test_subgen_api_internal_from_imports_resolve() -> None:
    module_to_exports: dict[str, set[str]] = {}
    for py_file in (SRC_ROOT / "subgen").rglob("*.py"):
        module_to_exports[_module_name(py_file)] = _module_exports(py_file)

    problems: list[str] = []
    for py_file in API_SRC.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        current = _module_name(py_file)
        pkg = current.split(".")[:-1]

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue

            if node.level > 0:
                base = pkg[: len(pkg) - node.level + 1]
                module = ".".join(base + [node.module])
            else:
                module = node.module

            if not module.startswith("subgen.api"):
                continue

            exported = module_to_exports.get(module)
            if exported is None:
                continue

            for alias in node.names:
                if alias.name == "*":
                    continue
                if alias.name in exported:
                    continue

                submodule = f"{module}.{alias.name}"
                if submodule in module_to_exports:
                    continue

                problems.append(
                    f"{py_file.relative_to(ROOT)}:{node.lineno} imports missing symbol "
                    f"{alias.name!r} from {module!r}"
                )

    assert not problems, "\n".join(problems)
