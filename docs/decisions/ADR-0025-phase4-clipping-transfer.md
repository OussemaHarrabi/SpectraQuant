# A-0025: frozen transfer study and the SpectraQuant research narrative

12 October 2026. Frozen before new cloud execution. User authorized continuing
the presented plan, with a coherent storyline and useful measured interpretation.

## Narrative and questions

The initial low-rank/bit allocation method did not beat uniform INT4. Correcting
the deployed residual operator did not rescue the additive calibration surrogate.
Phase3 then measured optimistic joint predictions and calibration-transfer failure;
a simpler conventional clipping control improved one fresh holdout by1.93%.
The next question is whether that simpler local criterion transfers across
architectures, scale and text domains. This extends the same compression study;
it does not retrospectively complete the original Tier2 or claim new algorithms.

## Models and corpora

- SmolLM2-360M @f8027fd0eaeea54caa13c31d31b9fdc459c38b49 (Apache2.0).
- Pythia-410M @9879c9b5f8bea9051dcb0e68dff21493d67e9d4f (Apache2.0;
  nominal checkpoint name; distinguish floating parameters from boolean buffers
  in Hub tensor counts, and record actual named-parameter count after loading).
- C4 @1588ec454efa1a09f29cd18ddd04fe05fc8653a2: train calibration,
  validation rows320:640 reserved for this study, disjoint from Phase3's first160.
- PG19 @4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b: first50 test books,
  Apache2.0 dataset card. Loader storage headers checked without local text access.

PTB was only a candidate in the prior proposal; its card leaves terms underspecified.
Use PG19 and the reserved unseen C4 document slice instead. No raw evaluation text
is inspected locally or used to select thresholds. Pretraining exposure is unknown.
Dataset revision verification alone does not pin externally hosted payload bytes;
record full document fingerprints and require equal fingerprints across both runs.

## Frozen arms and comparisons

INT4, symmetric, group32, original inference FP16. All linear weights excluding
weights tied to embeddings; unique unchanged parameters retained. Alpha grid
{1,.95,.9,.85,.8,.75}, ties retain the earliest option (alpha1 is first).

- RTN axis0 and unweighted MSE clipping axis0: primary transfer pair.
- RTN axis1 and unweighted MSE clipping axis1: grouping-orientation control.
- Diagonal-activation-weighted MSE clipping axis1, calibration subsets0/1/2:
  secondary criterion and calibration-robustness study, not full AWQ.
- FP16 quality reference, outside compressed-storage comparisons.

Axis0 groups lie along output rows in each input column, so a diagonal input
second moment merely rescales each group's MSE and cannot change its minimizer.
The activation-weighted criterion is therefore tested only on axis1, with its
own same-axis RTN and MSE comparators. No confounded attribution of an axis
change to activation weighting. Three subsets are not training seeds.

Capture FP16 input second moments from16x256-token blocks per subset, sampled
without replacement from a32,768-token pool of the first200 C4 train documents.
Use identical source document fingerprints across models, tokenizer-specific blocks.
No gradient training, NLL-based threshold search or holdout selection.

All seven compressed checkpoints are written, then reloaded before evaluation.
Persist a frozen descriptor containing their checksums before holdout text access.
Evaluate every predeclared arm; do not choose a headline arm from test performance.

## Evaluation and interpretation

Keep document boundaries; at most2,048 input tokens per document, sequence1024,
exactly51,200 predicted tokens per corpus/model. Refuse an insufficient corpus or
fewer than20 evaluated documents. Report token counts and document fingerprints.
Within each model/corpus all arms use identical blocks. Tokenizer-dependent raw
perplexities are not comparable between models; compare paired log ratios only.

Primary endpoint: axis0 MSE clipping vs axis0 RTN in four model/corpus cells.
Transfer gate: positive reduction in at least3/4 cells and unweighted geometric
mean perplexity reduction >=0.5%. No revision after results. Report each cell,
including regressions. Secondary axes/activation comparisons are exploratory.
Paired document-cluster bootstrap is descriptive corpus-sampling uncertainty,
not independent model/training replication or publication-level significance.

Measure target payload and full checkpoint bytes separately, verify all hashes.
Same-axis pairs must have identical payload and full checkpoint size. Cross-axis
comparisons must disclose measured layout overhead and satisfy the existing0.5%
full-byte comparison tolerance; otherwise exclude an equal-memory claim.
Class2 dense reconstructed quality; class3 actual stored files; no GPU speed claim.

## Resource and stopping rules

Two sequential private/free T4x2 jobs, each platform cap5,400s (90min), scientific
runner4,200s (70min): maximum6 visible-device-hours for the complete study.
Both configs freeze before the first result. No setting changes between runs.
Preserve failures and partial evidence; no automatic retry. Early build/evaluation
timing gates and disk checks must leave enough export/evaluation margin.

The user confirmed20h before earlier work. Reserving both prior full caps costs
9.333h, plus this6h leaves4.667h, above the3h reserve. Any optional kernel study
must fit a further1.5h cap; it is gated on transfer, compatibility and available
quota. No promise to run it if it requires substantial backend changes.

Publish the coherent diagnosis, transfer results and limitations regardless of
sign. A positive conventional control does not validate the original allocator.
Package a technical report and research CV evidence; no SOTA/acceptance claim.
