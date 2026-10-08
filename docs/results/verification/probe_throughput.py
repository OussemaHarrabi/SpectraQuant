"""Adversarial verification probe E4 (Agent L): is the Tier-1 'train from scratch, >=5 seeds'
cell executable on this CPU-only box? Measures real training throughput, no repo imports.
"""
import time
import torch
import torch.nn as nn

torch.set_num_threads(8)
print("torch", torch.__version__, "threads", torch.get_num_threads())

torch.manual_seed(0)


class Block(nn.Module):
    def __init__(self, d, nhead):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        a = torch.softmax((q @ k.transpose(-2, -1)) / C ** 0.5
                          + torch.triu(torch.full((T, T), float("-inf")), 1), dim=-1)
        x = x + self.proj(a @ v)
        return x + self.fc2(torch.relu(self.fc1(self.ln2(x))))


class TinyLM(nn.Module):
    def __init__(self, vocab, d, L):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, 4) for _ in range(L)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx, targets):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.lnf(x))
        return nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1))


def bench(vocab, d, L, seq, batch, steps=6, warmup=2):
    m = TinyLM(vocab, d, L)
    nparam = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    idx = torch.randint(0, vocab, (batch, seq))
    tgt = torch.randint(0, vocab, (batch, seq))
    for i in range(warmup + steps):
        if i == warmup:
            t0 = time.perf_counter()
        loss = m(idx, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    dt = time.perf_counter() - t0
    tok = steps * batch * seq
    return nparam, tok / dt, dt


print(f"{'config':>28} {'params(M)':>10} {'tok/s':>10} {'h per 1e9 tok':>14} {'h per 1 epoch WikiText-2 (2.4M tok)':>36}")
rows = []
for (vocab, d, L, seq, batch) in ((50257, 128, 4, 256, 8),
                                  (50257, 256, 4, 256, 8),
                                  (50257, 256, 6, 512, 8),
                                  (50257, 384, 6, 512, 4)):
    n, tps, dt = bench(vocab, d, L, seq, batch)
    rows.append((f"vocab{vocab} d{d} L{L} seq{seq} b{batch}", n / 1e6, tps))
    print(f"{rows[-1][0]:>28} {n/1e6:>10.1f} {tps:>10.1f} {1e9/tps/3600:>14.1f} {2.4e6/tps/3600:>36.2f}")

print()
print("Extrapolation for the >=5-seed Tier-1 requirement (preregistration SS6, SS8 rows 2/7):")
for name, npm, tps in rows:
    per_epoch_h = 2.4e6 / tps / 3600
    for epochs in (1, 5):
        for arms in (6, 12):
            tot = per_epoch_h * epochs * arms * 5
            print(f"  {name}: {epochs} epoch(s) x {arms} arms x 5 seeds = {tot:.0f} h = {tot/24:.1f} days")
