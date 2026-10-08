"""Adversarial verification probe E3 v2 (Agent L) -- proxy composition and activation shift.

Question attacked (docs/coordination/design-m2-interfaces.md invariant 3):
  proxy cost = E_X ||X W^T - X (Q(B)Q(A))^T||^2, "aggregated across layers by summation unless a
  slice documents otherwise".  Does that layerwise, summed quantity track downstream damage when
  layers are compressed JOINTLY?

Measured on a real decoder-only transformer (6 blocks, d=192, 24 linear layers) run in fp64 on CPU,
optionally trained first on a peaked Markov corpus so that softmax/logits are non-degenerate.

Outputs:
  A. per-layer single-compression damage vs that layer's proxy  -> gain spread, Spearman
  B. joint compression: measured damage vs (i) sum of single-layer damage, (ii) summed proxy
  C. activation shift: proxy recomputed on the compressed model's in-situ activations
Everything is deterministic; no repo module is imported.
"""
import math
import time
import torch
import torch.nn as nn

torch.manual_seed(20261008)
torch.set_num_threads(8)
DT = torch.float64
VOCAB, D, L, SEQ, BATCH = 512, 192, 6, 64, 6


class Block(nn.Module):
    def __init__(self, d, use_ln=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d) if use_ln else nn.Identity()
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d) if use_ln else nn.Identity()
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        a = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(C)
                          + torch.triu(torch.full((T, T), float("-inf")), 1), dim=-1)
        x = x + self.proj(a @ v)
        return x + self.fc2(torch.relu(self.fc1(self.ln2(x))))


class TinyLM(nn.Module):
    def __init__(self, vocab, d=192, layers=6, use_ln=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, use_ln) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d) if use_ln else nn.Identity()
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))

    def hidden(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.lnf(x)


def quant_int4(w, g=32):
    out, inn = w.shape
    nb = inn // g
    Wr = w.reshape(out, nb, g)
    s = Wr.abs().amax(dim=2, keepdim=True) / 7.0
    s = torch.where(s == 0, torch.full_like(s, 1e-12), s)
    return (torch.clamp(torch.round(Wr / s), -8, 7) * s).reshape(out, inn)


def truncate(w, r):
    u, sv, vh = torch.linalg.svd(w, full_matrices=False)
    return (u[:, :r] * sv[:r]) @ vh[:r]


def linears(model):
    return [(f"b{i}.{n}", getattr(b, n))
            for i, b in enumerate(model.blocks) for n in ("qkv", "proj", "fc1", "fc2")]


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        for pos, i in enumerate(order):
            rk[i] = pos
        return rk
    x, y = rank(a), rank(b)
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = math.sqrt(sum((p - mx) ** 2 for p in x) * sum((q - my) ** 2 for q in y))
    return num / den


def markov_corpus(n_tokens, seed=3):
    """Peaked transition matrix -> a trained model develops sharp softmax/attention."""
    g = torch.Generator().manual_seed(seed)
    P = torch.eye(VOCAB, dtype=DT) * 0.02 + torch.rand((VOCAB, VOCAB), generator=g, dtype=DT)
    P = P ** 4
    P = P / P.sum(-1, keepdim=True)
    out = torch.zeros(n_tokens, dtype=torch.long)
    out[0] = torch.randint(0, VOCAB, (1,), generator=g)
    for t in range(1, n_tokens):
        out[t] = torch.multinomial(P[out[t - 1]], 1, generator=g)
    return out


def train(model, steps=300, seq=SEQ, batch=BATCH, lr=3e-3):
    data = markov_corpus(200_000)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(11)
    model.train()
    for s in range(steps):
        ix = torch.randint(0, len(data) - seq - 1, (batch,), generator=g)
        x = torch.stack([data[i:i + seq] for i in ix])
        y = torch.stack([data[i + 1:i + seq + 1] for i in ix])
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return float(loss)


def run(use_ln=True, train_steps=300, r=32, b=4, tag=""):
    gen = torch.Generator().manual_seed(7)
    eval_idx = torch.randint(0, VOCAB, (BATCH, SEQ), generator=gen)
    # a held-out Markov sample so the model sees in-distribution text
    ev = markov_corpus(BATCH * (SEQ + 1), seed=99).reshape(BATCH, SEQ + 1)
    eval_idx = ev[:, :-1]
    tgt = ev[:, 1:]

    model = TinyLM(VOCAB, D, L, use_ln).to(DT)
    if train_steps:
        final = train(model, train_steps)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    lins = linears(model)

    def forward_with_inputs():
        inputs = {}
        hs = [lin.register_forward_pre_hook(
            lambda m, a, n=name: inputs.__setitem__(n, a[0].detach())) for name, lin in lins]
        with torch.no_grad():
            logits = model(eval_idx)
        for h in hs:
            h.remove()
        return logits, inputs

    ref_logits, fp_inputs = forward_with_inputs()
    with torch.no_grad():
        ref_hidden = model.hidden(eval_idx)

    def kl(a, b):
        pa, pb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
        return float((pa.exp() * (pa - pb)).sum(-1).mean())

    # ---- A: compress exactly one linear at a time
    single = {}
    for name, lin in lins:
        W = lin.weight.data
        Wq = quant_int4(truncate(W, r), 32) if b == 4 else truncate(W, r)
        delta = W - Wq
        X = fp_inputs[name]
        proxy = float(((X @ delta.T) ** 2).sum(-1).mean())
        orig = lin.weight.data.clone()
        lin.weight.data = Wq
        with torch.no_grad():
            h = model.hidden(eval_idx)
            lg = model(eval_idx)
        lin.weight.data = orig
        single[name] = dict(proxy=proxy,
                            d_hidden=float(((h - ref_hidden) ** 2).sum(-1).mean()),
                            d_logits=float(((lg - ref_logits) ** 2).sum(-1).mean()),
                            kl=kl(ref_logits, lg))

    # ---- B/C: compress everything; in-situ activations
    saved = {n: l.weight.data.clone() for n, l in lins}
    deltas = {}
    for name, lin in lins:
        Wq = quant_int4(truncate(lin.weight.data, r), 32) if b == 4 else truncate(lin.weight.data, r)
        deltas[name] = (lin.weight.data - Wq).clone()
        lin.weight.data = Wq
    joint_logits, situ_inputs = forward_with_inputs()
    with torch.no_grad():
        joint_hidden = model.hidden(eval_idx)
    for name, lin in lins:
        lin.weight.data = saved[name]

    joint_d_hidden = float(((joint_hidden - ref_hidden) ** 2).sum(-1).mean())
    joint_kl = kl(ref_logits, joint_logits)
    sum_single = sum(v["d_hidden"] for v in single.values())
    sum_proxy = sum(v["proxy"] for v in single.values())
    situ = {n: float(((situ_inputs[n] @ deltas[n].T) ** 2).sum(-1).mean()) for n in situ_inputs}
    shift = {n: situ[n] / single[n]["proxy"] - 1 for n in situ}

    order = list(single)
    print("\n" + "=" * 112)
    print(f"[{tag}] use_ln={use_ln} train_steps={train_steps} rank={r} bits={b} "
          f"n_linears={len(order)}  final train loss={final if train_steps else float('nan'):.3f}")
    print("=" * 112)
    print(f"{'layer':>9} {'proxy(fpX)':>11} {'proxy(in-situ)':>15} {'shift%':>9} {'d_hidden':>11} "
          f"{'d_logits':>11} {'KL':>10} {'gain':>9}")
    for n in order:
        v = single[n]
        print(f"{n:>9} {v['proxy']:>11.5g} {situ[n]:>15.5g} {100*shift[n]:>9.2f} "
              f"{v['d_hidden']:>11.5g} {v['d_logits']:>11.5g} {v['kl']:>10.4g} "
              f"{v['d_hidden']/v['proxy']:>9.4g}")
    gains = [single[n]["d_hidden"] / single[n]["proxy"] for n in order]
    rho_pd = spearman([single[n]["proxy"] for n in order],
                      [single[n]["d_hidden"] for n in order])
    rho_sd = spearman([situ[n] for n in order], [single[n]["d_hidden"] for n in order])
    rho_pp = spearman([single[n]["proxy"] for n in order], [situ[n] for n in order])
    shift_ratio = sum(situ.values()) / sum_proxy
    print(f"\n  Spearman(proxy on fp activations , single-layer downstream damage) = {rho_pd:.4f}")
    print(f"  Spearman(proxy on in-situ activs , single-layer downstream damage) = {rho_sd:.4f}")
    print(f"  Spearman(proxy fp activs         , proxy in-situ activs)           = {rho_pp:.4f}")
    print(f"  downstream gain spread: min={min(gains):.4g}  max={max(gains):.4g}  ratio={max(gains)/min(gains):.1f}x")
    print(f"  sum of single-layer damages = {sum_single:.6g}")
    print(f"  MEASURED joint damage (all layers compressed) = {joint_d_hidden:.6g}   joint KL = {joint_kl:.4g}")
    print(f"  superposition ratio  joint / sum-single = {joint_d_hidden/sum_single:.4f}")
    print(f"  sum of layerwise proxies (design-note aggregate) = {sum_proxy:.6g}")
    print(f"  joint damage / summed proxy = {joint_d_hidden/sum_proxy:.4f}")
    print(f"  summed proxy after activation shift = {sum(situ.values()):.6g}  (x{shift_ratio:.4f} vs fp-X)")
    return dict(rho_pd=rho_pd, rho_sd=rho_sd, rho_pp=rho_pp, gains=gains,
                joint=joint_d_hidden, sum_single=sum_single, sum_proxy=sum_proxy,
                shift_ratio=shift_ratio, kl=joint_kl)


t0 = time.perf_counter()
res = {}
for tag, (use_ln, steps, r, b) in {
    "trained-LN-r32-b4": (True, 300, 32, 4),
    "trained-noLN-r32-b4": (False, 300, 32, 4),
    "trained-LN-r64-b4": (True, 300, 64, 4),
    "trained-LN-r32-b8": (True, 300, 32, 8),
    "untrained-LN-r32-b4": (True, 0, 32, 4),
}.items():
    res[tag] = run(use_ln, steps, r, b, tag)

print("\n" + "=" * 112)
print("SUMMARY")
print("=" * 112)
print(f"{'config':>20} {'rho(proxy,damage)':>18} {'rho(in-situ,damage)':>20} {'joint/sum_single':>17} "
      f"{'joint/sum_proxy':>15} {'proxy shift x':>14} {'gain spread':>12}")
for k, o in res.items():
    print(f"{k:>20} {o['rho_pd']:>18.4f} {o['rho_sd']:>20.4f} {o['joint']/o['sum_single']:>17.4f} "
          f"{o['joint']/o['sum_proxy']:>15.4f} {o['shift_ratio']:>14.4f} "
          f"{max(o['gains'])/min(o['gains']):>11.1f}x")
print(f"\nwall {time.perf_counter()-t0:.1f}s")
