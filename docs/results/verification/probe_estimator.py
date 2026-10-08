"""Adversarial verification probe E2 (Agent L): the calibration-sample estimator of the proxy unit.

Proxy unit (docs/coordination/design-m2-interfaces.md invariant 3):
    E_X ||X W^T - X (Q(B)Q(A))^T||^2 = E_X ||x^T Delta||^2 = tr(Delta Sigma_X Delta^T),
Delta = W - Q(B)Q(A) fixed.  Sample estimator T_hat_n = (1/n) sum_i ||x_i^T Delta||^2.

Questions attacked: is the estimator biased, and how large is its variance?
Vectorised quadratic form (x^T G x with x = Lc z) so the probe runs in seconds, not minutes.
"""
import numpy as np

rng = np.random.default_rng(20261008)
print("numpy", np.__version__)
d = 256


def quant_int4_pergroup(W, g=32, sym=True):
    out, inn = W.shape
    nb = inn // g
    Wr = W.reshape(out, nb, g)
    if sym:
        s = np.abs(Wr).max(axis=2, keepdims=True) / 7.0
        s = np.where(s == 0, 1e-12, s)
        Wq = np.clip(np.round(Wr / s), -8, 7) * s
    else:
        mn, mx = Wr.min(axis=2, keepdims=True), Wr.max(axis=2, keepdims=True)
        s = np.where((mx - mn) == 0, 1e-12, (mx - mn) / 15.0)
        z = np.round(-mn / s)
        Wq = (np.clip(np.round(Wr / s + z), 0, 15) - z) * s
    return Wq.reshape(out, inn)


W = rng.standard_normal((d, d))
Wq = quant_int4_pergroup(W, 32, sym=True)
Delta = W - Wq
G = Delta @ Delta.T


def cov_gauss():
    A = rng.standard_normal((d, d)) * 0.3 + np.eye(d)
    return A @ A.T / d


def cov_heavy():
    scales = np.exp(rng.standard_normal(d) * 1.2)      # ~1e4 dynamic range across channels
    A = rng.standard_normal((d, d)) * 0.3 + np.eye(d)
    return np.diag(scales) @ (A @ A.T / d) @ np.diag(scales)


def simulate(Gtil, n, reps, chunk=25):
    est = np.empty(reps)
    for b in range(0, reps, chunk):
        k = min(chunk, reps - b)
        Z = rng.standard_normal((k, n, d))
        Y = Z @ Gtil
        est[b:b + k] = (Y * Z).sum(axis=(1, 2)) / n
    return est


print("=" * 104)
print("E2a  plug-in estimator: bias and variance for the SINGLE-LAYER, ORIGINAL-ACTIVATION target")
print("=" * 104)
print("Exact relative SE for Gaussian activations: sqrt(2*tr(Gtil^2)/n)/tr(Gtil).")
print(f"{'regime':>22} {'n':>7} {'truth':>12} {'exact rel SE':>13} {'sim rel SE':>11} {'sim rel bias':>13}")

for name, covf in (("gaussian", cov_gauss), ("heavy-tailed (LLM-like)", cov_heavy)):
    Sigma = covf()
    Lc = np.linalg.cholesky(Sigma + 1e-9 * np.eye(d))
    truth = float(np.trace(G @ Sigma))
    Gtil = Lc.T @ G @ Lc
    exact = float(np.sqrt(2 * np.trace(Gtil @ Gtil)) / np.trace(Gtil))
    for n in (32, 128, 512, 2048, 8192):
        reps = 2000 if n <= 512 else 300
        est = simulate(Gtil, n, reps)
        print(f"{name:>22} {n:>7} {truth:>12.5g} {exact/np.sqrt(n):>13.4f} "
              f"{est.std()/est.mean():>11.4f} {est.mean()/truth-1:>13.5f}")
    print(f"{'':>22} {'exact rel SE at n=128 / 524288 (the predeclared calibration):':>40} "
          f"{exact/np.sqrt(128):.4f} / {exact/np.sqrt(524288):.2e}")

print()
print("=" * 104)
print("E2b  bias from fitting the operator on the SAME activations used to score it (AWQ-style)")
print("=" * 104)
print(f"{'n':>7} {'in-sample mean':>16} {'held-out mean':>16} {'underestimate factor':>21}")
for n in (128, 512, 2048):
    ins, outs = [], []
    for _ in range(300):
        X = rng.standard_normal((n, d))
        s = np.abs(X).mean(axis=0) ** 0.5
        s = s / s.mean()
        Wqs = quant_int4_pergroup(W * s[None, :], 32, True) / s[None, :]
        Gi = (W - Wqs) @ (W - Wqs).T
        ins.append(float((X @ Gi * X).sum(axis=1).mean()))
        Xh = rng.standard_normal((n, d))
        outs.append(float((Xh @ Gi * Xh).sum(axis=1).mean()))
    print(f"{n:>7} {np.mean(ins):>16.6g} {np.mean(outs):>16.6g} {np.mean(outs)/np.mean(ins):>21.4f}")
