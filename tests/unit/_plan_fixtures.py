"""Shared builders for the ``test_plan_*`` modules (not a test module itself).

Nothing here touches the network or a GPU: the Hugging Face model/tokenizer/dataset stack is
replaced by deterministic stand-ins so the *plan runner's own* code paths — arm resolution, arm
application, the revision gate, manifest construction — are what the tests exercise.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn

#: Vocabulary of the tiny LM stand-in.
VOCAB = 11

#: The revision the frozen plans pin for their model (``HuggingFaceTB/SmolLM2-135M``).
PIN = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"

#: A different, equally valid-looking revision (for the mismatch test).
OTHER_PIN = "fedcba9876543210fedcba9876543210fedcba98"

#: The model the frozen plans pin.
PLAN_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"


class TinyLM(nn.Module):
    """A deterministic causal-LM stand-in with two untied linear projections and a head."""

    def __init__(
        self,
        *,
        vocab: int = VOCAB,
        width: int = 8,
        context: int = 16,
        commit: str | None = None,
        seed: int = 7,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab, width)
        self.proj = nn.Linear(width, width)
        self.head = nn.Linear(width, vocab)
        config = SimpleNamespace(max_position_embeddings=context)
        if commit is not None:
            config._commit_hash = commit  # the attribute Hugging Face sets on a load
        self.config = config

    def forward(self, ids: Tensor) -> Any:
        hidden = torch.tanh(self.proj(self.embed(ids)))
        return SimpleNamespace(logits=self.head(hidden))


class TinyTokenizer:
    """A deterministic byte-ish tokenizer stand-in (no ``transformers`` import)."""

    def __init__(self, *, vocab: int = VOCAB, tokens_per_document: int = 12) -> None:
        self.vocab = vocab
        self.tokens_per_document = tokens_per_document
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __call__(self, text: str, return_tensors: str = "pt") -> Any:
        assert return_tensors == "pt", "the runner only requests 'pt'"
        length = max(2, min(self.tokens_per_document, len(text.split()) or 2))
        ids = torch.arange(length, dtype=torch.long).unsqueeze(0) % self.vocab
        return SimpleNamespace(input_ids=ids)


def documents(count: int = 3, words: int = 12) -> list[str]:
    """Return deterministic document texts for the perplexity corpus."""
    return [" ".join(f"token{index}" for index in range(words)) for _ in range(count)]


def fake_transformers(commit: str | None = None) -> SimpleNamespace:
    """Return a stand-in for the ``transformers`` module used by ``plan_runner.load_model``."""

    class _ModelLoader:
        @staticmethod
        def from_pretrained(identifier: str, **kwargs: Any) -> TinyLM:
            context = int(kwargs.pop("context", 16))
            return TinyLM(context=context, commit=commit)

    class _TokenizerLoader:
        @staticmethod
        def from_pretrained(identifier: str, **kwargs: Any) -> TinyTokenizer:
            return TinyTokenizer()

    return SimpleNamespace(
        AutoModelForCausalLM=_ModelLoader(),
        AutoTokenizer=_TokenizerLoader(),
    )


def _fixture_training_corpus(
    config: Any,
    tokenizer: Any,
    *,
    seq_len: int,
    max_documents: Any,
    context: Any = None,
    device: Any = "cpu",
):
    """A fixture-sized training corpus: ``TinyLM`` has a context of 16, so windows are 16 wide."""
    from spectraquant.training.loop import SequenceData

    width = 16
    generator = torch.Generator().manual_seed(int(config.seeds.master))
    # Token ids must stay inside the fixture vocabulary, or the embedding lookup indexes out of range.
    train = torch.randint(0, VOCAB, (8, width), generator=generator, dtype=torch.int64)
    val = torch.randint(0, VOCAB, (2, width), generator=generator, dtype=torch.int64)
    provenance = {
        "training.train_dataset": "fixture",
        "training.val_dataset": "fixture",
        "training.train_split": "fixture",
        "training.val_split": "fixture",
        "training.train_windows": int(train.shape[0]),
        "training.val_windows": int(val.shape[0]),
        "training.seq_len": width - 1,
        "training.corpus_checksum": "sha256:fixture",
        "training.split_rule": (
            "fixture stand-in: the plan's training.seq_len exceeds the fixture model's context of 16, "
            "so the offline corpus is synthetic and 16 tokens wide; real runs use the plan's pinned "
            "corpus and width"
        ),
    }
    return SequenceData(train=train, val=val, checksum="sha256:fixture"), provenance


def plan_runner_env(monkeypatch: Any, *, commit: str | None = PIN) -> None:
    """Patch the model/dataset loaders so ``run_plan`` runs offline and deterministically."""
    from spectraquant.cloud import plan_data, plan_runner

    monkeypatch.setattr(plan_data, "require_transformers", lambda: fake_transformers(commit))
    monkeypatch.setattr(plan_runner, "require_transformers", lambda: fake_transformers(commit))
    monkeypatch.setattr(
        plan_runner,
        "verify_pinned_revision",
        lambda repo_id, revision, *, repo_type: revision,
    )
    monkeypatch.setattr(
        plan_runner,
        "load_plan_texts",
        lambda ref, **kwargs: documents(),
    )
    # The trainable arms need a corpus, and the fixture model's context is 16, so the pinned
    # `training.seq_len` cannot be used here. The substitution is recorded in the provenance rather
    # than hidden: the manifest must not claim a window width the fixture did not build.
    monkeypatch.setattr(plan_runner, "_training_corpus", _fixture_training_corpus)
