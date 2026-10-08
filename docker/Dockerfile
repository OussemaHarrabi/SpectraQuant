# syntax=docker/dockerfile:1
#
# CPU-only image for SpectraQuant. It exists so that (a) Linux tooling and the CPU PyTorch wheel
# resolve identically to the pinned lockfile and (b) the build itself is a smoke test: if the tiny
# end-to-end experiment or its manifest validation fails, the image does not build.
#
#   docker build -f docker/Dockerfile -t spectraquant:cpu .
#   docker run --rm spectraquant:cpu smoke --config configs/experiment/smoke.yaml --out /tmp/out
#
# No CUDA base image, no GPU runtime, no CUDA wheels: the research path is CPU-only by contract
# (AGENTS.md section 2, ADR-0002).

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    SPECTRAQUANT_REPO_ROOT=/app

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /uvx /usr/local/bin/

WORKDIR /app

# Only what the runtime needs: the lockfile, the package, the configs, the schema and the committed
# sample manifest. Docs, tests and reports stay out of the image.
COPY pyproject.toml uv.lock README.md LICENSE AGENTS.md ./
COPY src/ src/
COPY configs/ configs/
COPY artifacts/ artifacts/
COPY scripts/ scripts/

RUN uv sync --frozen

# Build-time proof: run the tiny experiment and validate the manifest it wrote.
RUN uv run --no-sync spectraquant smoke --config configs/experiment/smoke.yaml --out /tmp/docker-smoke --quiet \
      > /tmp/smoke-manifest-path.txt \
 && uv run --no-sync spectraquant validate-manifest "$(cat /tmp/smoke-manifest-path.txt)"

RUN useradd --create-home --uid 10001 spectraquant \
 && chown -R spectraquant:spectraquant /app /opt/venv
USER spectraquant

ENTRYPOINT ["uv", "run", "--no-sync", "spectraquant"]
CMD ["--help"]
