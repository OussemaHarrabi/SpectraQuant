# Design note — Milestone 2 cross-slice interfaces (frozen)

Status: **frozen by the orchestrator** before any Milestone 2 implementation. Slices (quantization,
factorization, proxy, allocation) may not change a signature listed here without an amendment to this
file plus a status entry. Rationale: the scientific claims depend on these four slices sharing one
memory model and one factor convention; divergence would silently invalidate equal-memory comparisons.

## 0. Hard invariants

1. **One byte-accounting source of truth.** Allocation, training, conversion and reporting MUST all
   obtain costs from `spectraquant.quantization.accounting`. No other module may re-derive bytes.
2. **One factorization convention.** `W ≈ B @ A` with `A: (rank, in_features)`, `B: (out_features, rank)`,
   `W: (out_features, in_features)`. Documented once in `factorization/svd.py`, referenced everywhere.
3. **One proxy unit.** Proxy cost is the per-layer squared Frobenius norm of the *layer output* error
   `E_X ||X W^T − X (Q(B) Q(A))^T||²`, aggregated across layers by summation unless a slice documents
   otherwise. Weight-space quantities are converted into this unit before comparison, or reported
   separately and never mixed.
4. **Approximation labelling.** Every proxy returns `exact: bool` and `measurement_class`; an
   estimator MUST also return its sampling information (sample count, dtype) so variance is reportable.
5. **No CUDA / no third-party quantized kernels** in `src/spectraquant/**` core paths. CUDA-gated code
   may exist only behind an explicit, documented guard that raises.
6. **Docstrings**: shapes, dtypes, device assumptions, numerical limitations on every public function
   (AGENTS.md §7). Pure-torch + numpy only in M2 slices.

## 1. `spectraquant.quantization` (owner D)

```python
@dataclass(frozen=True)
class QuantSpec:
    bits: int                     # 2,3,4,8 supported; validated
    granularity: str              # "per_tensor" | "per_channel" | "per_group"
    group_size: int | None        # required iff granularity == "per_group"
    symmetric: bool
    axis: int = 0                 # channel axis for per_channel/per_group
    round_mode: str = "nearest"   # "nearest" only in M2

def quant_params(w: Tensor, spec: QuantSpec) -> QuantParams          # scales, zero_points, qmin, qmax
def fake_quantize(w: Tensor, spec: QuantSpec, params: QuantParams | None = None) -> Tensor
def fake_dequantize(q: Tensor, params: QuantParams, spec: QuantSpec) -> Tensor
def exact_round_trip_ok(w: Tensor, spec: QuantSpec) -> bool           # dequant(quant(w)) == w for representable w

def pack_int4(q: Tensor) -> Tensor          # signed int4 -> uint8 pairs, little-nibble-first
def unpack_int4(packed: Tensor, shape, axis_padding: int) -> Tensor
def serialized_size_bytes(w_shape, spec: QuantSpec) -> int            # from accounting, not from tensors
def theoretical_bits(w_shape, spec: QuantSpec) -> float               # analytical estimate (class 1)
def accounted_bytes(w_shape, spec: QuantSpec) -> int                  # includes scales, zero points,
                                                                      # group metadata, padding, alignment
def measure_serialized_bytes(state: dict[str, Tensor], specs: dict[str, QuantSpec]) -> int  # class 3
```

Rules: `theoretical_bits` is class 1 (analytical) and MUST NOT be presented as measured storage.
`accounted_bytes` must equal `measure_serialized_bytes` for the formats we serialize, within an
explicitly documented alignment rule — asserted by test. Group metadata overhead is counted, never
omitted. Non-divisible group sizes must be handled and tested.

## 2. `spectraquant.factorization` (owner E)

```python
def truncated_svd(w: Tensor, rank: int, *, energy_keep: float | None = None) -> LowRankFactors  # (A, B)
def randomized_svd(w: Tensor, rank: int, *, n_oversamples: int, n_iter: int, seed: int) -> LowRankFactors
def reconstruction_error(w: Tensor, f: LowRankFactors, *, norm: str = "fro") -> float
def singular_values(w: Tensor, *, top_k: int | None = None) -> Tensor
def effective_rank(w: Tensor, *, energy: float = 0.99) -> float
def stable_rank(w: Tensor) -> float
def spectral_summary(w: Tensor) -> dict[str, float]   # must include the summaries the proxy consumes
def factor_bytes(in_features: int, out_features: int, rank: int, dtype_bytes: int = 2) -> int  # class 1
```

`LowRankFactors` carries `A`, `B`, `rank`, `dtype`, and the reconstruction error at construction.
Randomized approximations MUST be validated against exact SVD on small matrices with a reported bound.

## 3. `spectraquant.proxies` (owner F)

```python
@dataclass(frozen=True)
class ProxyResult:
    value: float                  # squared-Frobenius output error per invariant 3
    exact: bool
    measurement_class: int        # 1 analytical, 2 fake-quantization-derived
    per_layer: dict[str, float]
    diagnostics: dict[str, float] # e.g. sample count, condition number, sampled variance

class Proxy(Protocol):
    name: str
    def score_layer(self, w: Tensor, x: Tensor, rank: int, spec: QuantSpec) -> ProxyResult: ...
    def score_model(self, layers: Mapping[str, LayerInputs], ranks: Mapping[str, int],
                    specs: Mapping[str, QuantSpec]) -> ProxyResult: ...
```

Required implementations: `weight_frobenius` (baseline), `activation_weighted`, `hessian_diag`
(or `fisher_diag` when `X^T X` is the surrogate), `quant_residual_stats`, `spectral_summary`-based,
and `combined` (the candidate; weights must be explicit config values, never hard-coded).

Ground truth for validation: `spectraquant.evaluation.toy.exact_output_error(w, x, factors, spec)`
computed in float64 by brute force, with the sample-level and mean-level variants distinguished.

## 4. `spectraquant.allocation` (owner H)

```python
@dataclass(frozen=True)
class AllocationProblem:
    layer_shapes: Mapping[str, tuple[int, int]]
    ranks: Sequence[int]                 # allowed ranks
    bits: Sequence[int]                  # allowed bit widths
    budget_bytes: int
    cost_fn: Callable[[str, int, int], int]     # MUST be quantization.accounting-backed
    error_fn: Callable[[str, int, int], float]  # MUST be proxy-backed
    overhead_fn: Callable[[str, int, int], int] = ...  # residual/outlier/alignment extras

@dataclass(frozen=True)
class Allocation:
    per_layer: Mapping[str, tuple[int, int]]   # (rank, bits)
    accounted_bytes: int
    predicted_error: float
    solver: str
    feasible: bool
    diagnostics: dict[str, float]

def solve_uniform(problem) -> Allocation
def solve_greedy(problem) -> Allocation          # marginal error reduction per byte, deterministic
def solve_exhaustive(problem) -> Allocation      # tiny models only; oracle
def solve_ortools(problem, *, time_limit_s: float, seed: int) -> Allocation  # CP-SAT, optional extra
```

Rules: infeasibility raises `InfeasibleBudget` carrying the minimum feasible budget; `solve_exhaustive`
is the oracle and MUST agree with `solve_ortools` on tiny instances (test); allocations are emitted as
deterministic JSON manifests with the git SHA, the cost model version, and the byte total, and must be
reproducible from the manifest alone.

## 5. Test-file ownership (no collisions)

| Slice | Own test files |
|---|---|
| D | `tests/unit/test_quant_*.py`, `tests/integration/test_pack_roundtrip.py` |
| E | `tests/unit/test_factorization_*.py`, `tests/unit/test_spectral_*.py` |
| F | `tests/unit/test_proxy_*.py`, `tests/unit/test_toy_ground_truth.py` |
| H | `tests/unit/test_allocation_*.py`, `tests/unit/test_memory_accounting.py` |

Shared fixtures live in `tests/fixtures/` and are owned by whoever creates them first; additions must
be appended, never edited in place by another slice.

## 6. Verification gate for M2

1. All M2 unit tests green on CPU.
2. `accounted_bytes == measure_serialized_bytes` for every serialized format (test).
3. Proxy vs brute-force float64 ground truth: relative error within a documented tolerance, with the
   tolerance justified (not chosen to pass).
4. `solve_exhaustive == solve_ortools` objective on at least 3 tiny instances; greedy reported but not
   required to match.
5. No CUDA import in `src/spectraquant/**` (checked by a test).
