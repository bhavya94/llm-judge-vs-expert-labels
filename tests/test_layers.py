"""The package layers the README describes: each package imports only the packages below it, and only the CLI prints."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "llm_judge_vs_expert_labels"
BELOW = {
    "data": set(),
    "labeling": {"data"},
    "judges": {"data"},
    "evaluation": set(),
}


def _trees():
    return {path: ast.parse(path.read_text()) for path in SRC.rglob("*.py")}


def _imported_packages(tree: ast.AST):
    """The top-level llm_judge_vs_expert_labels package of every import; "." for a relative import, which isn't allowed."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                yield "."
            elif (node.module or "").startswith("llm_judge_vs_expert_labels."):
                yield node.module.split(".")[1]
        elif isinstance(node, ast.Import):
            yield from (alias.name.split(".")[1] for alias in node.names
                        if alias.name.startswith("llm_judge_vs_expert_labels."))


def test_each_package_imports_only_the_packages_below_it():
    wrong = []
    for path, tree in _trees().items():
        package = path.relative_to(SRC).parts[0]
        allowed = BELOW.get(package, set(BELOW)) | {package}
        wrong += [f"{path.relative_to(SRC)} imports {imported}" for imported in _imported_packages(tree)
                  if imported not in allowed]
    assert wrong == []


def test_only_the_cli_prints():
    printing = [str(path.relative_to(SRC)) for path, tree in _trees().items() if path.name != "cli.py"
                and any(isinstance(n, ast.Call) and getattr(n.func, "id", None) == "print" for n in ast.walk(tree))]
    assert printing == []
