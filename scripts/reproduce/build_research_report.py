"""Compile the publication PDF directly from the canonical LaTeX manuscript."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports" / "paper" / "spectraquant-research-paper.tex"
OUTPUT = ROOT / "output" / "pdf" / "SpectraQuant_Research_Report_2026.pdf"


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _resolve_engine(explicit: str | None) -> tuple[str, Path]:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"LaTeX engine does not exist: {path}")
        kind = "tectonic" if path.name.lower().startswith("tectonic") else "latexmk"
        return kind, path

    for kind, name in (("tectonic", "tectonic"), ("latexmk", "latexmk")):
        resolved = shutil.which(name)
        if resolved:
            return kind, Path(resolved)

    raise RuntimeError(
        "No LaTeX engine found. Install Tectonic or latexmk, or pass "
        "--engine-path C:/path/to/tectonic.exe."
    )


def build(engine_path: str | None = None) -> Path:
    """Build the tracked PDF from SOURCE without maintaining parallel report prose."""
    kind, engine = _resolve_engine(engine_path)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="spectraquant-latex-") as raw_tmp:
        tmp = Path(raw_tmp)
        if kind == "tectonic":
            _run(
                [
                    str(engine),
                    "-X",
                    "compile",
                    "--keep-logs",
                    "--outdir",
                    str(tmp),
                    str(SOURCE),
                ],
                cwd=SOURCE.parent,
            )
        else:
            _run(
                [
                    str(engine),
                    "-pdf",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    f"-output-directory={tmp}",
                    str(SOURCE),
                ],
                cwd=SOURCE.parent,
            )

        compiled = tmp / f"{SOURCE.stem}.pdf"
        if not compiled.is_file():
            raise FileNotFoundError(f"Compiler did not produce {compiled}")
        shutil.copy2(compiled, OUTPUT)

    return OUTPUT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile the canonical SpectraQuant LaTeX manuscript."
    )
    parser.add_argument(
        "--engine-path",
        help="Optional explicit path to tectonic or latexmk.",
    )
    args = parser.parse_args()
    result = build(args.engine_path or os.environ.get("SPECTRAQUANT_LATEX_ENGINE"))
    print(result)


if __name__ == "__main__":
    main()
