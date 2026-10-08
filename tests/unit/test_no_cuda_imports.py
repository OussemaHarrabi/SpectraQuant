"""Repo-wide invariant: no CUDA-only dependency anywhere in the core package.

``docs/coordination/design-m2-interfaces.md`` section 6 gate 5 and ``AGENTS.md`` section 2 require
that no file under ``src/spectraquant/**`` (except the ``cloud/`` adapter package, whose imports are
gated by design) imports a CUDA-only library or moves a tensor to CUDA. This module scans the source
text with :mod:`ast` — not a grep — so comments and docstrings that merely *mention* CUDA are not
flagged, while every real import and call site is.

The scan is self-tested against planted violations, so a broken pattern cannot make the suite pass
silently.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "spectraquant"
#: Packages deliberately excluded from the invariant (see docs/coordination/ownership.md, stream K).
EXCLUDED_PACKAGES = frozenset({"cloud"})
#: Top-level modules that are CUDA-only for our purposes (AGENTS.md section 5, class 4-GPU).
CUDA_ONLY_MODULES = frozenset(
    {
        "awq",
        "auto_gptq",
        "bitsandbytes",
        "deepspeed",
        "exllamav2",
        "flash_attn",
        "gptqmodel",
        "marlin",
        "mslk",
        "triton",
        "vllm",
    }
)
#: Modules whose *import* means CUDA-gated code (attribute probes such as torch.cuda.is_available()
#: are documented guards and stay allowed by design note invariant 5).
CUDA_GATED_MODULES = frozenset({"torch.cuda"})


def _scanned_files() -> list[Path]:
    """Every ``*.py`` file of the core package, excluding the documented exclusions."""
    files: list[Path] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative_parts = path.relative_to(PACKAGE_ROOT).parts
        if EXCLUDED_PACKAGES & set(relative_parts):
            continue
        files.append(path)
    return files


def _is_cuda_module(name: str) -> bool:
    """True when ``name`` is a CUDA-only module or a CUDA-gated torch submodule path."""
    root = name.split(".")[0]
    return root in CUDA_ONLY_MODULES or any(
        name == gated or name.startswith(f"{gated}.") for gated in CUDA_GATED_MODULES
    )


def _string_mentions_cuda(node: ast.AST) -> bool:
    """True when ``node`` is a string literal naming a CUDA device."""
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and "cuda" in node.value


def _violations(source_path: Path) -> list[str]:
    """Return one human-readable line per CUDA violation found in ``source_path``."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_cuda_module(alias.name):
                    found.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _is_cuda_module(module) or (
                module == "torch" and any(alias.name == "cuda" for alias in node.names)
            ):
                found.append(f"line {node.lineno}: from {module} import ...")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "cuda":
                found.append(f"line {node.lineno}: .cuda() call")
            elif node.func.attr == "to" and any(_string_mentions_cuda(arg) for arg in node.args):
                found.append(f"line {node.lineno}: .to('cuda') call")
            for keyword in node.keywords:
                if keyword.arg == "device" and _string_mentions_cuda(keyword.value):
                    found.append(f"line {node.lineno}: device='cuda' argument")
    return found


def test_no_cuda_imports_or_calls_in_the_core_package() -> None:
    """The real invariant: the core package is CUDA-free, file by file."""
    files = _scanned_files()
    violations = {str(path.relative_to(REPO_ROOT)): _violations(path) for path in files}
    offenders = {path: lines for path, lines in violations.items() if lines}
    assert not offenders, f"CUDA-only usage found in the core package: {offenders}"


def test_scan_covers_the_package_but_not_the_cloud_adapter() -> None:
    """The scan must actually visit files (a broken glob would make the invariant vacuous)."""
    files = _scanned_files()
    names = {path.name for path in files}
    assert len(files) >= 20, f"only {len(files)} files scanned: the glob is broken"
    assert "fake_quant.py" in names
    assert "packing.py" in names
    assert "accounting.py" in names
    assert not any("cloud" in path.relative_to(PACKAGE_ROOT).parts for path in files)


def test_scanner_detects_each_planted_violation(tmp_path: Path) -> None:
    """Prove the detector is not vacuous: every violation class is caught in a planted file."""
    planted = tmp_path / "planted.py"
    planted.write_text(
        "\n".join(
            [
                "import bitsandbytes as bnb",  # line 1
                "import vllm",  # line 2
                "import torch.cuda",  # line 3
                "from torch.cuda.amp import autocast",  # line 4
                "from torch import cuda",  # line 5
                "t = torch.zeros(2).cuda()",  # line 6
                "t = torch.zeros(2).to('cuda')",  # line 7
                "t = torch.zeros(2, device='cuda:0')",  # line 8
                "import torchao",  # line 9 (allowed)
            ]
        ),
        encoding="utf-8",
    )
    reported = _violations(planted)
    assert len(reported) == 8
    assert any("import bitsandbytes" in line for line in reported)
    assert any("import vllm" in line for line in reported)
    assert any("from torch.cuda" in line for line in reported)
    assert any("from torch import" in line for line in reported)
    assert any(".cuda() call" in line for line in reported)
    assert any(".to('cuda') call" in line for line in reported)
    assert any("device='cuda' argument" in line for line in reported)
    assert not any("torchao" in line for line in reported)


def test_scan_ignores_mentions_in_comments_and_docstrings(tmp_path: Path) -> None:
    """A docstring that mentions CUDA without importing it is not a violation."""
    benign = tmp_path / "benign.py"
    benign.write_text(
        '"""No CUDA here: bitsandbytes and vllm are deliberately absent."""\n'
        "# torch.cuda.is_available() is a documented guard elsewhere, not an import.\n"
        "VALUE = 'cuda'\n",
        encoding="utf-8",
    )
    assert _violations(benign) == []
