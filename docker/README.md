# Docker image (CPU only)

`docker/Dockerfile` builds a CPU-only image of the repository. It is the Linux execution path for
tooling that has no Windows support, and it doubles as a build-time smoke test.

## Build and run

Run from the **repository root** (the build context must include `pyproject.toml` and `uv.lock`):

```bash
docker build -f docker/Dockerfile -t spectraquant:cpu .
docker run --rm spectraquant:cpu --help
docker run --rm spectraquant:cpu env
docker run --rm spectraquant:cpu smoke --config configs/experiment/smoke.yaml --out /tmp/out
docker run --rm spectraquant:cpu validate-manifest /tmp/out/smoke-*.manifest.json
```

The image installs exactly what `uv.lock` pins (`uv sync --frozen`), so the Linux dependency set
cannot drift from the committed lockfile. Only runtime dependencies are installed — the `dev` and
`track` extras are not part of the image.

## What the build proves

The `RUN` step after `uv sync` executes the tiny end-to-end experiment and validates the manifest it
wrote. A broken config, a nondeterministic or crashing training loop, or a manifest that no longer
matches `artifacts/schemas/run-manifest.schema.json` fails the build.

## Boundaries

* **No GPU.** The image has no CUDA runtime, no CUDA wheels and no ROCm. Measurement classes 4 and 5
  remain unavailable (see `docs/research/environment.md` and ADR-0002); nothing in the image may be
  used to produce latency or kernel-backed numbers.
* **No data or weights.** Models are defined in code (`configs/model/*.yaml` + the tiny transformer)
  and the Tier-0/1 corpus is generated from a seeded recurrence. Nothing is downloaded at build
  time, so the image is reproducible offline once the lockfile is resolved.
* **Artifacts.** Run outputs written under `/tmp` are discarded with the container; mount a volume
  and pass `--out` if a manifest must be kept.

## Multi-arch note

The PyTorch CPU index publishes wheels for `linux/amd64` and `linux/arm64`. On Apple Silicon or
Windows, build for the matching platform explicitly, e.g. `--platform linux/amd64`.
