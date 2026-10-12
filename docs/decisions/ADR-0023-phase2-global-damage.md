# ADR-0023: score the exact stored intervention, permit plain quantization

2026-10-11. Accepted under the user's approved continuation, 20 Kaggle GPU-hours remaining.

The frozen proxy computes a factorization of W; the deployed method adds a quantized
residual to a quantized base. Keep that historical experiment unchanged. Introduce
a separate exploratory module and CLI command. Score Q(W)+Q(B)Q(A), with Q defined
by serialization and reload, in a model with a uniform-INT4 background. Include
rank zero so the allocator can choose to spend no bytes on a residual.

CP-SAT solves the additive marginal-NLL surrogate. Marginal damages can be
negative; do not clip them. Interactions make additivity an approximation.
Compare bits-only and bits-plus-residual allocation at the same measured target
payload ceiling as uniform INT4. Save full checkpoint file sizes separately.
All target payloads are exact packed blobs; quality is float16 reconstruction,
class 2. There is no low-bit runtime or latency claim.

Three calibration subsets test allocation stability, not training randomness.
C4 calibrates, WikiText validation selects a family with minimum mean gain
0.005 NLL, then every frozen arm is evaluated on WikiText test exactly once.
The test was used in historical phases: this is exploratory evidence, not a
new blind confirmation. A new model/benchmark is required for generalization.

One private generated Kaggle notebook, 12,000 s platform limit, 9,500 s runner
limit with an additional 1,200 s preflight reserve. At two GPUs this bounds the
first attempt to 6.67 device-hours including installation/export. Keep at least
3 hours unused overall. No training, paid compute, or quota-expanding sweep.
Preserve failure logs, partial scores and manifests. No repeated submission of
an active notebook. Commit and push code before submitting an immutable SHA.
