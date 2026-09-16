"""The generation-sanity probe: rung 4's T10 producer.

T10 exists because loss can look tolerable while generation collapses, so the
probe must measure the *generated text* rather than a perplexity. These tests
pin the properties that make the verdict trustworthy:

- termination, cap-hit, repetition and looping are measured, not inferred;
- the prompt set is hash-checked, so "the same probe" means the same prompts;
- an empty prompt set is refused rather than reporting healthy ratios over no
  evidence;
- decoding settings are the declared ones, applied identically to both arms.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from chowder.backends.router_healing import RouterHealingEvalSpec
from chowder.backends.router_healing_eval_worker import (
    _distinct_trigram_ratio,
    _generation_sanity,
    _looping_completions,
    _read_generation_prompts,
)

#: Token ids the fake tokenizer can turn back into words.
_WORDS = {10: "alpha", 11: "beta", 12: "gamma", 13: "delta"}


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, return_tensors=None, add_special_tokens=None):
        return {"input_ids": torch.tensor([[1, 1]])}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(_WORDS[int(i)] for i in ids if int(i) in _WORDS)


class _ScriptedLm:
    """Returns one scripted continuation per successive prompt."""

    def __init__(self, scripts: list[list[int]]) -> None:
        self._scripts = list(scripts)
        self.calls = 0
        self.training = True

    def generate(self, ids, max_new_tokens=None, **kwargs):
        script = self._scripts[self.calls % len(self._scripts)]
        self.calls += 1
        fresh = torch.tensor([script], dtype=ids.dtype)
        return torch.cat([ids, fresh], dim=1)

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self


def _eval_spec(**overrides) -> RouterHealingEvalSpec:
    base = {
        "base_model_dir": "F:/models/base",
        "base_content_sha256": "a" * 64,
        "payload_dir": None,
        "holdout_corpus_path": "F:/data/holdout.txt",
        "holdout_corpus_sha256": "b" * 64,
        "expected_parameter_paths": (),
        "output_dir": "F:/out",
        "seq_len": 8,
        "batches": 2,
    }
    base.update(overrides)
    return RouterHealingEvalSpec(**base)


# --- the pure metrics -------------------------------------------------------


def test_repetition_lowers_the_distinct_trigram_ratio():
    varied = ["alpha beta gamma delta epsilon"]
    repetitive = ["alpha alpha alpha alpha alpha"]
    assert _distinct_trigram_ratio(varied) == 1.0
    assert _distinct_trigram_ratio(repetitive) < 0.5


def test_no_trigrams_is_unmeasured_not_a_perfect_score():
    """An empty completion must not read as 100% diverse."""
    assert _distinct_trigram_ratio([]) is None
    assert _distinct_trigram_ratio(["alpha beta"]) is None


def test_a_completion_with_three_consecutive_identical_trigrams_loops():
    assert _looping_completions(["alpha alpha alpha alpha alpha"]) == 1
    assert _looping_completions(["alpha alpha alpha beta gamma"]) == 0
    assert _looping_completions(["alpha beta gamma", "x y z w"]) == 0


def test_looping_counts_only_the_looping_completions():
    texts = ["a a a a a", "b c d e f", "z z z z z"]
    assert _looping_completions(texts) == 2


# --- the prompt set is pinned ----------------------------------------------


def _prompts_file(tmp_path: Path, body: str) -> tuple[Path, str]:
    path = tmp_path / "prompts.txt"
    path.write_text(body, encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_prompts_are_read_without_blanks_or_comments(tmp_path: Path):
    path, digest = _prompts_file(tmp_path, "# a comment\n\nfirst prompt\nsecond prompt\n")
    spec = _eval_spec(
        generation_probe={
            "prompts_path": str(path),
            "prompts_sha256": digest,
            "max_new_tokens": 4,
            "temperature": 0.7,
            "top_p": 0.95,
        }
    )
    assert _read_generation_prompts(spec) == ["first prompt", "second prompt"]


def test_a_drifted_prompt_set_is_refused(tmp_path: Path):
    path, _digest = _prompts_file(tmp_path, "first prompt\n")
    spec = _eval_spec(
        generation_probe={
            "prompts_path": str(path),
            "prompts_sha256": "c" * 64,
            "max_new_tokens": 4,
            "temperature": 0.7,
            "top_p": 0.95,
        }
    )
    with pytest.raises(RuntimeError, match="prompts hash mismatch"):
        _read_generation_prompts(spec)


def test_an_empty_prompt_set_is_refused(tmp_path: Path):
    path, digest = _prompts_file(tmp_path, "# only a comment\n\n")
    spec = _eval_spec(
        generation_probe={
            "prompts_path": str(path),
            "prompts_sha256": digest,
            "max_new_tokens": 4,
            "temperature": 0.7,
            "top_p": 0.95,
        }
    )
    with pytest.raises(RuntimeError, match="prompt set is empty"):
        _read_generation_prompts(spec)


# --- the probe's own measurement -------------------------------------------


def test_the_probe_measures_termination_cap_repetition_and_looping():
    lm = _ScriptedLm(
        [
            [10, 11, 12, 2],  # ends on eos -> terminated
            [10] * 6,  # repeats to the cap: 4 identical trigrams -> looping
            [10, 11, 12, 13, 10, 11],  # hits the cap without looping
        ]
    )
    probe = {"max_new_tokens": 6, "temperature": 0.7, "top_p": 0.95, "prompts_sha256": "d" * 64}
    result = _generation_sanity(
        torch, lm, _FakeTokenizer(), torch.device("cpu"), probe, ["p1", "p2", "p3"]
    )

    assert result["status"] == "measured"
    assert result["prompts"] == 3
    assert result["termination_rate"] == pytest.approx(1 / 3)
    assert result["max_token_cap_rate"] == pytest.approx(2 / 3)
    assert result["looping_prompts"] == 1
    # 9 trigrams across the three completions; (alpha,beta,gamma) appears in
    # both the first and third, so 5 of the 9 are distinct
    assert result["distinct_trigram_ratio"] == pytest.approx(5 / 9)
    # compression ratio is reported, never gated
    assert result["compression_ratio"] is not None
    # and the task score is honestly unmeasured rather than invented
    assert result["task_score"]["status"] == "UNMEASURED"
    assert len(result["completions"]) == 3


def test_the_probe_records_the_exact_settings_it_used():
    lm = _ScriptedLm([[10, 11, 12, 2]])
    probe = {"max_new_tokens": 7, "temperature": 0.25, "top_p": 0.5, "prompts_sha256": "e" * 64}
    result = _generation_sanity(
        torch, lm, _FakeTokenizer(), torch.device("cpu"), probe, ["only"]
    )
    assert result["max_new_tokens"] == 7
    assert result["temperature"] == 0.25
    assert result["top_p"] == 0.5
    assert result["prompts_sha256"] == "e" * 64


def test_the_probe_restores_training_mode():
    lm = _ScriptedLm([[10, 11, 12, 2]])
    lm.train()
    probe = {"max_new_tokens": 4, "temperature": 0.7, "top_p": 0.95, "prompts_sha256": "f" * 64}
    _generation_sanity(torch, lm, _FakeTokenizer(), torch.device("cpu"), probe, ["one"])
    assert lm.training is True


# --- the spec refuses an unpinnable probe ----------------------------------


def test_a_probe_without_a_prompt_hash_is_refused():
    with pytest.raises(ValueError, match="prompts_sha256 must be a sha256 hex digest"):
        _eval_spec(generation_probe={"prompts_path": "F:/p.txt", "max_new_tokens": 4})


def test_a_probe_with_an_out_of_range_top_p_is_refused():
    with pytest.raises(ValueError, match=r"top_p must be in \(0, 1\]"):
        _eval_spec(
            generation_probe={
                "prompts_path": "F:/p.txt",
                "prompts_sha256": "a" * 64,
                "max_new_tokens": 4,
                "temperature": 0.7,
                "top_p": 1.5,
            }
        )


def test_a_probe_with_a_non_positive_max_new_tokens_is_refused():
    with pytest.raises(ValueError, match="max_new_tokens must be a positive integer"):
        _eval_spec(
            generation_probe={
                "prompts_path": "F:/p.txt",
                "prompts_sha256": "a" * 64,
                "max_new_tokens": 0,
                "temperature": 0.7,
                "top_p": 0.95,
            }
        )


def test_a_probe_with_a_non_positive_temperature_is_refused():
    with pytest.raises(ValueError, match="temperature must be positive"):
        _eval_spec(
            generation_probe={
                "prompts_path": "F:/p.txt",
                "prompts_sha256": "a" * 64,
                "max_new_tokens": 4,
                "temperature": 0.0,
                "top_p": 0.95,
            }
        )


def test_no_declared_probe_leaves_the_spec_valid():
    """The probe is opt-in; existing evaluations must not become invalid."""
    assert _eval_spec().generation_probe is None
