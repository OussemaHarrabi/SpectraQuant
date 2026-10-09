# Model card — tiny fixture transformer (in-repo, two configurations)

> Pinned by `configs/model/tiny.yaml` (the CI smoke model) and by the `TinyConfig` dataclass in
> `src/spectraquant/evaluation/toy.py` (the fixture behind the committed proxy and allocation
> artifacts). Card author: M10 release slice. Values are quoted from those two sources.

## Identity

- **Model id / role:** `tiny-char-transformer` (`configs/model/tiny.yaml`) — the sanctioned local CI
  smoke fixture; and `TinyConfig`/`TinyTransformer` (`src/spectraquant/evaluation/toy.py`) — the
  Tier-0 evaluation fixture.
- **Revision:** the repository commit that contains this card; there is no external checkpoint. The
  fixture is reconstructed from the config + a fixed seed, so it is reproducible bit-for-bit on CPU.
- **License:** our own artifact, Apache-2.0 (this repository's license). No third-party weights.
- **Size:** smoke config `vocab_size: 32`, `d_model: 64`, `n_heads: 4`, `n_layers: 2`,
  `max_seq_len: 32`, `dropout: 0.0` (≤ 200 k parameters by construction,
  `src/spectraquant/training/tiny_lm.py`). Evaluation fixture `TinyConfig` defaults:
  `n_blocks: 3`, `d_model: 48`, `n_heads: 2`, `d_ff: 96`, `vocab: 24`, `seq_len: 24`,
  `use_norm: True`, `final_norm: True`, `seed: 0` (13 compressible linear modules).

## Intended use

- The CI smoke run (`uv run spectraquant smoke --config configs/experiment/smoke.yaml`): a harness
  self-test that produces a schema-valid manifest and is bit-for-bit reproducible at a fixed seed.
- Tier-0 fixtures: exact per-layer damage, proxy-vs-exact float64 agreement, allocator oracle checks
  and the proxy-ranking sweep. Every committed `LOCAL-FIXTURE` artifact comes from `TinyConfig`.

## Out-of-scope use

- **It is not a language model.** It is trained on a deterministic synthetic next-token task
  (`configs/data/synthetic.yaml`, an LCG recurrence) and says nothing about LM compression.
- No claim may be sourced from it as confirmatory: it is exploratory by construction
  (`preregistration.md` §8 marks the Tier-0 cells exploratory).

## Status

**MEASURED at Tier-0 fixture scale only, with honest limits.** The proxy fixture, the proxy
validation sweep, the allocation frontier and the regularizer sweep are all measured on this fixture
(classes 1–3, one seed or five seeds). Unlike the pinned large models, this one *has* produced
SpectraQuant evidence — but that evidence is about a tiny, briefly-trained synthetic model and is
**not** evidence about transformers at any real scale. The confirmatory Tier-1 vehicle (pinned
`L_b = 8`, `d = 256`, 32 M-parameter transformer) is **NOT RUN**.

## Limitations

- Tiny vocabulary (24 or 32) and sequence length (24 or 32): no long-context or rare-token behaviour.
- Randomly initialised and briefly trained (120 steps for the evaluation fixture; 50 for the smoke
  config): its spectra and activation statistics are those of an undertrained model.
- The two configurations differ (48/3-block evaluation fixture vs 64/2-layer smoke model); quote the
  one you mean and cite the config path.

## Provenance

- Configs: `configs/model/tiny.yaml`, `configs/data/synthetic.yaml`, `configs/experiment/smoke.yaml`,
  `configs/experiment/regularizer_tier0.yaml`.
- Code: `src/spectraquant/training/tiny_lm.py`, `src/spectraquant/evaluation/toy.py`.
- Artifacts: `artifacts/sample-results/{smoke-manifest.json, proxy-fixture/, proxy-validation/,
  allocation-frontier/, regularizer/}`.
