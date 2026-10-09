#!/usr/bin/env python
"""Write a tiny local Hugging Face model + tokenizer, for offline ``run-plan`` path checks.

``spectraquant run-plan`` normally loads the plan's pinned model and pinned dataset. Nothing about
that can be exercised on the workstation (``AGENTS.md`` §2.3 forbids research inference locally and
there is no network access in CI), so the sanctioned rehearsal is a *substituted* run: a tiny model
directory and a small corpus, both recorded in the manifest as stand-ins that make the run **not** a
plan measurement (``--allow-local-substitution``).

Usage::

    uv run python scripts/cloud/make_tiny_model.py /tmp/sq-tiny
    printf 'the quick brown fox\\n\\nspectraquant measures low rank and low bit\\n' > /tmp/sq-corpus.txt
    uv run spectraquant run-plan --plan configs/tier1/smollm2_135m.yaml \\
        --arms ptq_uniform_4 --model-dir /tmp/sq-tiny --perplexity-text /tmp/sq-corpus.txt \\
        --allow-local-substitution --out /tmp/sq-run --seq-len 64

The model is a 2-layer, 32-wide Llama (a few hundred kB) with an untied head, so the linear-weight
selection, the arm transforms and the manifest path are all real; only the weights and the corpus are
not the plan's.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_VOCAB = 256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dest", type=Path, help="Directory to write the tiny model into.")
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB, help="Vocabulary size.")
    args = parser.parse_args()

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    lines = [
        "the quick brown fox jumps over the lazy dog .",
        "spectraquant studies low rank and low bit compression for transformers .",
    ] * 40
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        lines,
        trainer=trainers.BpeTrainer(
            vocab_size=args.vocab, special_tokens=["<eos>"], show_progress=False
        ),
    )
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<eos>")
    config = LlamaConfig(
        vocab_size=fast.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    args.dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.dest)
    fast.save_pretrained(args.dest)
    print(f"wrote {model.num_parameters()} parameters (vocab {fast.vocab_size}) to {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
