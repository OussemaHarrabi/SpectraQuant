# Security policy

## Scope

SpectraQuant is a research repository: it trains small models on synthetic data and writes JSON
manifests. It runs offline, on CPU, and never ships model weights or datasets. The realistic
security surface is therefore small but not empty:

* **Deserialization.** `torch.load` and similar loaders can execute arbitrary code when given an
  untrusted checkpoint. The project loads no external checkpoints today; if you add a loader, treat
  every checkpoint as untrusted, prefer `weights_only=True`, and document the risk.
* **Dependency compromise.** The dependency set is pinned in `uv.lock`; the advisory scan in
  `.github/workflows/security.yml` is expected to be read before any tagged release.
* **Secrets.** No credentials, tokens or private keys may be committed. The `detect-private-key`
  pre-commit hook and the repository `.gitignore` are the first line of defence; the remote
  `security` workflow is the second.
* **Untrusted input to the CLI.** `spectraquant validate-manifest` only reads and validates JSON —
  it never executes manifest content.

## Reporting a vulnerability

Report privately through GitHub's
[security advisories](https://github.com/OussemaHarrabi/SpectraQuant/security/advisories/new) (or
email the maintainer listed in `CITATION.cff`). Do not open a public issue for a vulnerability.

Please include:

* the affected commit (`git rev-parse HEAD`) or release tag,
* a minimal reproduction (command, config, input file),
* the impact you believe it has, and whether untrusted input is required.

## Response expectations

This is a single-maintainer research project with no service deployment, so there is no SLA.
Reports are acknowledged as soon as possible; confirmed issues are fixed on a `fix/` branch and
noted in the release notes. There are no supported release versions yet — `main` is the only
branch, and no build is currently distributed to third parties.

## Out of scope

* Results being wrong in a research sense (that is a bug — open a normal issue with the manifest).
* Missing GPU acceleration, or the absence of measurement classes 4-5: these are documented hardware
  constraints, not vulnerabilities (see `docs/research/environment.md` and ADR-0002).
