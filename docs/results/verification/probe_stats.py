"""Adversarial verification probe E1/E2 (Agent L, read-only w.r.t. the repo) -- fast version.

E1: sample-size behaviour of the H2 correlation test when the resampling unit is the LAYER
    (preregistration SS7.2: "bootstrap CIs (per-layer resampling)").
E2: bias/variance of the calibration-sample estimator of the proxy unit
    E_X ||X W^T - X (Q(B)Q(A))^T||^2  (docs/coordination/design-m2-interfaces.md invariant 3).

Pure numpy. No repo module is imported or modified.
"""
import numpy as np

rng = np.random.default_rng(20261008)
print("numpy", np.__version__)
Z = 1.959963985


def ranks(v):
    return np.argsort(np.argsort(v)).astype(float)


def spearman_cols(A, y):
    """Spearman of every column of A (n x k) against y (n,)."""
    ra = ranks(y)
    return np.array([np.corrcoef(ranks(A[:, j]), ra)[0, 1] for j in range(A.shape[1])])


def fisher_ci(r, n):
    r = np.clip(r, -0.999999, 0.999999)
    zr = np.arctanh(r)
    se = 1.0 / np.sqrt(max(n - 3, 1))
    return np.tanh(zr - Z * se), np.tanh(zr + Z * se)


print("=" * 100)
print("E1  H2 correlation test with the LAYER as the resampling unit")
print("=" * 100)


def draw(n, rho, reps, seed):
    r = np.random.default_rng(seed)
    z = r.standard_normal((reps, n, 2))
    x = z[:, :, 0]
    y = np.exp(0.6 * (rho * z[:, :, 0] + np.sqrt(1 - rho ** 2) * z[:, :, 1]))
    return x, y


print(f"{'L':>3} {'rho_true':>9} {'mean_rho':>9} {'P(CI excl 0)':>13} {'Fisher CI width':>16} "
      f"{'layer-bootstrap CI width':>25}")
for n in (4, 6, 8, 12, 16, 24):
    for rho in (0.6, 0.8, 0.9):
        reps = 2000
        x, y = draw(n, rho, reps, 1000 + n)
        rs = np.array([spearman_cols(x[i][:, None], y[i])[0] for i in range(reps)])
        lo, hi = fisher_ci(rs, n)
        power = float(np.mean((lo > 0) | (hi < 0)))
        # declared unit: bootstrap over layers (preregistration SS7.2)
        bws = []
        for i in range(200):
            xb = x[i][:, None]
            bs = []
            for _ in range(200):
                idx = rng.integers(0, n, n)
                if len(set(idx.tolist())) < 4:
                    continue
                bs.append(spearman_cols(xb[idx], y[i][idx])[0])
            bs = np.array(bs)
            bws.append(np.percentile(bs, 97.5) - np.percentile(bs, 2.5))
        print(f"{n:>3} {rho:>9.2f} {rs.mean():>9.3f} {power:>13.3f} "
              f"{float((hi-lo).mean()):>16.3f} {float(np.mean(bws)):>25.3f}")

print()
print("Paired test: proxy (true rank corr 0.90) vs comparator (0.70) over L layers.")
print(f"{'L':>3} {'P(declared winner)':>19} {'mean dRho':>10} {'mean 95% CI width':>18}")
for n in (4, 6, 8, 12, 16, 24, 48):
    hits, ds, ws = 0, [], []
    reps = 2000
    r = np.random.default_rng(77000 + n)
    for s in range(reps):
        z = r.standard_normal((n, 4))
        # outcome depends on the shared factor z0 (plus independent noise)
        y = np.exp(0.6 * (0.85 * z[:, 0] + np.sqrt(1 - 0.7225) * z[:, 3]))
        a = 0.90 * z[:, 0] + np.sqrt(1 - 0.81) * z[:, 1]
        b = 0.70 * z[:, 0] + np.sqrt(1 - 0.49) * z[:, 2]
        ra, rb = spearman_cols(a[:, None], y)[0], spearman_cols(b[:, None], y)[0]
        ds.append(ra - rb)
        bs = []
        for _ in range(200):
            idx = r.integers(0, n, n)
            if len(set(idx.tolist())) < 4:
                continue
            bs.append(spearman_cols(a[idx, None], y[idx])[0] -
                      spearman_cols(b[idx, None], y[idx])[0])
        bs = np.array(bs)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ws.append(hi - lo)
        hits += int(lo > 0)
    print(f"{n:>3} {hits/reps:>19.3f} {np.mean(ds):>10.3f} {np.mean(ws):>18.3f}")

print()
print("=" * 100)
print("E2  calibration-sample estimator of the proxy unit")
print("=" * 100)


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


d = 256
r = np.random.default_rng(0)
W = r.standard_normal((d, d))
Wq = quant_int4_pergroup(W, 32, sym=True)
Delta = W - Wq
G = Delta @ Delta.T          # ||X Delta^T||^2 = x^T G x


def cov_gauss():
    A = r.standard_normal((d, d)) * 0.3 + np.eye(d)
    return A @ A.T / d


def cov_heavy():
    scales = np.exp(r.standard_normal(d) * 1.2)
    A = r.standard_normal((d, d)) * 0.3 + np.eye(d)
    return np.diag(scales) @ (A @ A.T / d) @ np.diag(scales)


for name, covf in (("gaussian", cov_gauss), ("heavy-tailed (LLM-like)", cov_heavy)):
    Sigma = covf()
    Lc = np.linalg.cholesky(Sigma + 1e-9 * np.eye(d))
    truth = float(np.trace(G @ Sigma))
    Gtil = Lc.T @ G @ Lc
    print(f"\n[{name}] true E||X.Delta^T||^2 = {truth:.6g}")
    print(f"{'n':>7} {'mean est':>16} {'rel bias':>10} {'rel SE':>10} {'rel SE of a k=6 layer ranking':>30}")
    for n in (32, 128, 512, 2048, 8192):
        reps = 4000
        Zx = r.standard_normal((reps, n, d))
        Xz = Zx @ Lc.T
        est = np.einsum('rnd,de,rne->r', Xz, G, Xz) / n
        print(f"{n:>7} {est.mean():>16.6g} {est.mean()/truth-1:>10.4f} "
              f"{est.std()/est.mean():>10.4f} {'-':>30}")

print("\n[in-sample bias] AWQ-style per-channel scales fitted on the SAME activations used to score")
print(f"{'n':>7} {'in-sample mean':>15} {'held-out mean':>15} {'underestimate factor':>21}")
for n in (128, 512, 2048):
    ins, outs = [], []
    for t in range(300):
        X = r.standard_normal((n, d))
        s = (np.abs(X).mean(axis=0) ** 0.5)
        s = s / s.mean()
        Wqs = quant_int4_pergroup(W * s[None, :], 32, True) / s[None, :]
        Gi = (W - Wqs) @ (W - Wqs).T
        ins.append(float(np.einsum('nd,de,ne->n', X, Gi, X).mean()))
        Xh = r.standard_normal((n, d))
        outs.append(float(np.einsum('nd,de,ne->n', Xh, Gi, Xh).mean()))
    print(f"{n:>7} {np.mean(ins):>15.6g} {np.mean(outs):>15.6g} {np.mean(outs)/np.mean(ins):>21.4f}")
