"""Batched evaluation must report what a single-row evaluation reports.

Batching is a throughput parameter, not a measurement one: with the dense
weights off the card the per-token weight re-stream dominates, so decoding
``batch_size`` rows in one ``generate`` call is what makes an arm measurement
finish at all (~15x here: 2.60 s/token for one row against 2.77 s/token for
sixteen). What must not change is what each row's generation is recorded as
having done -- the target instrument is defined over exactly those facts
(``generated_tokens``, ``eos_terminated``), and a batched run that misreported
them would move the finding rather than the cost.

The model here is a deterministic stand-in for ``generate``: it reproduces the
version's per-row finish semantics (a finished row's later steps are padded)
and derives each row's tokens from that row's *unpadded* prompt only, so a
single-row pass and a batched pass must agree row for row. The rules themselves
are pinned separately, without a model, in ``test_evaluator_generation``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from chowder.evaluators.transformers_text import (  # noqa: E402
    EvalSuiteSpec,
    TransformersTextEvalSpec,
)
from chowder.evaluators import transformers_text_worker as worker  # noqa: E402

PAD = 0
STOP = 1
#: Ids the stand-in "model" emits for a row that does not stop early. Kept well
#: clear of PAD/STOP so a real token can never be mistaken for padding.
_FIRST_REAL_TOKEN = 5


class _FakeTokenizer:
    """Enough tokenizer for the worker: left padding and a real decode rule."""

    def __init__(self, *args, **kwargs) -> None:
        self.pad_token_id = PAD
        self.eos_token_id = STOP
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.padding_side = "left"

    @staticmethod
    def _ids(text: str) -> list[int]:
        # Deterministic, and different per prompt: a row's generation depends on
        # its own prompt only, so both arms must produce the same row.
        return [((ord(character) % 6) + 2) for character in text] or [2]

    def __call__(self, texts, return_tensors=None, padding=False):
        assert return_tensors == "pt"
        rows = [self._ids(text) for text in texts]
        width = max(len(row) for row in rows) if len(rows) > 1 else len(rows[0])
        input_ids = []
        attention = []
        for row in rows:
            fill = width - len(row) if len(rows) > 1 else 0
            input_ids.append([self.pad_token_id] * fill + row)
            attention.append([0] * fill + [1] * len(row))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=False):
        special = {PAD, STOP}
        values = [int(token) for token in token_ids]
        if skip_special_tokens:
            values = [value for value in values if value not in special]
        return " ".join(str(value) for value in values)

    def apply_chat_template(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("these suites render raw prompts")


class _FakeModel:
    """``generate`` for a row whose tokens depend only on its own prompt."""

    def __init__(self, *args, stop_ids: list[int] | None = None, **kwargs) -> None:
        self._stop_ids = list(stop_ids or [STOP])
        self.generation_config = type(
            "Cfg", (), {"eos_token_id": self._stop_ids}
        )()
        self.config = type("Config", (), {"_commit_hash": None})()

    def eval(self):
        return self

    def parameters(self):
        return iter([torch.zeros(1)])

    def to(self, device):
        return self

    @staticmethod
    def _seed(prompt_ids):
        return int(sum(prompt_ids))

    def _next(self, seed: int, step: int) -> int:
        if seed % 2 == 0 and step == 2:
            # Half the rows stop early, half run into the cap: both branches of
            # the recording rule are exercised by the same run.
            return self._stop_ids[0]
        return _FIRST_REAL_TOKEN + ((seed + step) % 20)

    def generate(
        self,
        *,
        input_ids,
        attention_mask,
        max_new_tokens,
        pad_token_id,
        eos_token_id,
        do_sample=False,
        **kwargs,
    ):
        seeds = [
            self._seed(input_ids[row][attention_mask[row] == 1].tolist())
            for row in range(int(input_ids.shape[0]))
        ]
        finished = [False] * len(seeds)
        out = input_ids
        for step in range(max_new_tokens):
            if all(finished):
                break
            column = []
            for row, seed in enumerate(seeds):
                if finished[row]:
                    column.append(pad_token_id)
                    continue
                token = self._next(seed, step)
                if token in self._stop_ids:
                    finished[row] = True
                    # This version writes the terminator it produced and pads
                    # the row only from the next step on.
                    column.append(token)
                else:
                    column.append(token)
            out = torch.cat([out, torch.tensor(column, dtype=torch.long)[:, None]], dim=1)
        return out


def _stub(**factories):
    """A stand-in class whose ``from_pretrained`` returns the factory result."""
    return type(
        "Stub",
        (),
        {
            "from_pretrained": staticmethod(
                lambda *args, **kwargs: next(iter(factories.values()))(*args, **kwargs)
            )
        },
    )


def _suite(
    tmp_path: Path,
    rows: int,
    batch_size: int,
    max_new_tokens: int,
    n_samples: int = 1,
    temperature: float = 0.7,
    store_chains: bool = False,
) -> tuple:
    dataset = tmp_path / f"dataset-{rows}-{batch_size}.jsonl"
    dataset.write_text(
        "".join(
            json.dumps({"prompt": f"prompt-{index}-{'x' * index}", "expected": "5"}) + "\n"
            for index in range(rows)
        ),
        encoding="utf-8",
    )
    return EvalSuiteSpec(
        name="slice",
        dataset=str(dataset),
        prompt_field="prompt",
        expected_field="expected",
        scoring="normalized_exact_match",
        max_new_tokens=max_new_tokens,
        use_chat_template=False,
        batch_size=batch_size,
        n_samples=n_samples,
        temperature=temperature,
        store_chains=store_chains,
    )


def _run(
    tmp_path: Path,
    monkeypatch,
    *,
    rows: int,
    batch_size: int,
    max_new_tokens: int = 6,
    n_samples: int = 1,
    store_chains: bool = False,
):
    # The worker imports these inside evaluate(), so the patched owner is the
    # transformers module the import resolves against.
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    # Imported first, deliberately: importing peft *replaces* the transformers
    # entry in sys.modules, and a patch applied before it is silently discarded.
    # The worker then imports peft itself, which is already cached.
    pytest.importorskip("peft")
    import sys

    # The worker's `from transformers import ...` resolves through sys.modules,
    # so this is the object the patch must land on.
    transformers = sys.modules["transformers"]
    monkeypatch.setattr(
        transformers, "AutoTokenizer", _stub(AutoTokenizer=_FakeTokenizer)
    )
    monkeypatch.setattr(
        transformers,
        "AutoModelForCausalLM",
        _stub(AutoModelForCausalLM=lambda *a, **k: _FakeModel()),
    )
    assert sys.modules["transformers"].AutoTokenizer is not None
    output = tmp_path / f"out-{rows}-{batch_size}"
    spec = TransformersTextEvalSpec(
        base_model="fake-model",
        adapter_dir=None,
        output_dir=str(output),
        suites=(
            _suite(
                tmp_path,
                rows,
                batch_size,
                max_new_tokens,
                n_samples=n_samples,
                store_chains=store_chains,
            ),
        ),
        precision="bf16",
        quantization="none",
        device="cpu",
        placement="resident",
        seed=1234,
        offline=True,
    )
    return worker.evaluate(spec), output


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize("rows,batch_size", [(4, 1), (4, 4), (4, 3), (5, 2)])
def test_a_batched_run_reports_the_same_rows_as_a_single_row_run(
    tmp_path: Path, monkeypatch, rows: int, batch_size: int
) -> None:
    """Same model, same prompts, two throughputs: identical row facts."""
    _, single_dir = _run(tmp_path, monkeypatch, rows=rows, batch_size=1)
    _, batched_dir = _run(tmp_path, monkeypatch, rows=rows, batch_size=batch_size)

    single = _rows(single_dir / "predictions-slice.jsonl")
    batched = _rows(batched_dir / "predictions-slice.jsonl")

    assert len(single) == len(batched) == rows
    for index, (left, right) in enumerate(zip(single, batched)):
        assert left["prompt"] == right["prompt"]
        assert left["prediction"] == right["prediction"], f"row {index} differs"
        assert left["score"] == right["score"], f"row {index} scored differently"
        assert left["generated_tokens"] == right["generated_tokens"], f"row {index}"
        assert left["eos_terminated"] == right["eos_terminated"], f"row {index}"


def test_both_termination_kinds_are_actually_present_in_the_fixture(
    tmp_path: Path, monkeypatch
) -> None:
    """The equality above is only meaningful if both branches occur."""
    _, batched_dir = _run(tmp_path, monkeypatch, rows=4, batch_size=4)
    rows = _rows(batched_dir / "predictions-slice.jsonl")
    stopped = [row for row in rows if row["eos_terminated"]]
    capped = [row for row in rows if not row["eos_terminated"]]
    assert stopped and capped, rows
    assert all(row["generated_tokens"] < 6 for row in stopped)
    assert all(row["generated_tokens"] == 6 for row in capped)


def test_the_batch_size_the_run_used_is_recorded(tmp_path: Path, monkeypatch) -> None:
    payload, _ = _run(tmp_path, monkeypatch, rows=4, batch_size=3)
    assert payload["suites"]["slice"]["batch_size"] == 3


def test_two_throughputs_are_two_evaluation_specs(tmp_path: Path) -> None:
    """The spec a run filed names the throughput it was produced at."""

    def digest(batch_size: int) -> str:
        return TransformersTextEvalSpec(
            base_model="fake-model",
            adapter_dir=None,
            output_dir=str(tmp_path / f"d{batch_size}"),
            suites=(_suite(tmp_path, 4, batch_size, 6),),
            seed=1234,
        ).digest()

    assert digest(1) != digest(4)


# ---- self-consistency suite fields: identity + validation ----------------------


def _scoring_suite(**overrides):
    defaults = dict(
        name="s",
        dataset="d.jsonl",
        scoring="final_number_match",
        max_new_tokens=64,
    )
    return EvalSuiteSpec(**{**defaults, **overrides})


def test_sampling_fields_are_digest_additive_to_the_protocol_entry():
    from chowder.evaluators.transformers_text import suite_protocol_entry
    from chowder.protocol import protocol_fingerprint

    greedy = suite_protocol_entry(_scoring_suite(), "sha")
    assert "n_samples" not in greedy and "temperature" not in greedy
    sampled = suite_protocol_entry(_scoring_suite(n_samples=5, temperature=0.8), "sha")
    assert sampled["n_samples"] == 5 and sampled["temperature"] == 0.8
    assert protocol_fingerprint(greedy) != protocol_fingerprint(sampled)


def test_k1_suite_hashes_identically_regardless_of_temperature():
    """k=1 is greedy regardless of temperature, so identity must not move."""
    from chowder.evaluators.transformers_text import suite_protocol_entry

    assert suite_protocol_entry(_scoring_suite(temperature=0.9), "sha") == suite_protocol_entry(_scoring_suite(), "sha")


def test_n_samples_and_temperature_validation():
    with pytest.raises(ValueError):
        _scoring_suite(n_samples=0)
    with pytest.raises(ValueError):
        _scoring_suite(n_samples=4, temperature=0.0)
    with pytest.raises(ValueError):
        _scoring_suite(n_samples=4, temperature=-1.0)


def test_store_chains_records_sampled_chain_texts(tmp_path, monkeypatch):
    """store_chains persists each sampled chain's text for selection (RFT)."""
    from chowder.evaluators.transformers_text import SAMPLE_SEPARATOR

    _, out = _run(tmp_path, monkeypatch, rows=2, batch_size=1, n_samples=3, store_chains=True)
    rows = _rows(out / "predictions-slice.jsonl")
    assert len(rows) == 2
    for row in rows:
        assert len(row["chains"]) == 3
        assert all(isinstance(c, str) and c for c in row["chains"])
        assert row["prediction"] == SAMPLE_SEPARATOR.join(row["chains"])


def test_store_chains_defaults_to_absent(tmp_path, monkeypatch):
    """Without the flag, sampled rows carry no chain texts (artifact shape unchanged)."""
    _, out = _run(tmp_path, monkeypatch, rows=2, batch_size=1, n_samples=3)
    rows = _rows(out / "predictions-slice.jsonl")
    assert rows and all("chains" not in row for row in rows)


def test_store_chains_is_not_protocol_identity():
    """Recording changes what is stored, not what is scored: no digest movement."""
    from chowder.evaluators.transformers_text import suite_protocol_entry
    from chowder.protocol import protocol_fingerprint

    plain = suite_protocol_entry(_scoring_suite(), "sha")
    storing = suite_protocol_entry(_scoring_suite(store_chains=True), "sha")
    assert "store_chains" not in storing
    assert protocol_fingerprint(plain) == protocol_fingerprint(storing)
