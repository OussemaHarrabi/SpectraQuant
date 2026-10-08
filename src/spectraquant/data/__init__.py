"""Data package.

Tier 0 fixture data (implemented): deterministic synthetic corpora, :mod:`spectraquant.data.synthetic`.

Not implemented (Milestone 2, owner: data stream): pretrained tokenizers, WikiText-2 /
C4 / CIFAR pipelines, dataset checksum manifests, train/eval overlap detection. Those modules will
be added here; they must not be stubbed with fake data.
"""

from __future__ import annotations

from spectraquant.data.synthetic import SyntheticCorpus, build_corpus, sequence_checksum

__all__ = ["SyntheticCorpus", "build_corpus", "sequence_checksum"]
